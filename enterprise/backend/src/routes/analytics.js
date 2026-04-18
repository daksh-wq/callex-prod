import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/analytics
router.get('/', async (req, res) => {
    try {
        const { page = 1, limit = 50, sentiment, minDuration, disposition } = req.query;
        const pageNum = parseInt(page, 10);
        const limitNum = parseInt(limit, 10);

        let query = db.collection('calls').where('userId', '==', req.userId);
        const snap = await query.get();
        let calls = queryToArray(snap);
        calls.sort((a, b) => {
            const da = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
            const db2 = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
            return db2 - da;
        });

        // Apply filters in JS (Firestore doesn't support complex compound queries easily)
        if (sentiment) calls = calls.filter(c => c.sentiment === sentiment);
        if (minDuration) calls = calls.filter(c => (c.duration || 0) >= parseInt(minDuration, 10));
        if (disposition) calls = calls.filter(c => c.dispositionId === disposition);

        const total = calls.length;
        const paginated = calls.slice((pageNum - 1) * limitNum, pageNum * limitNum);

        // Enrich with agent/campaign data
        for (const call of paginated) {
            if (call.agentId) {
                const agentDoc = await db.collection('agents').doc(call.agentId).get();
                call.agent = agentDoc.exists ? docToObj(agentDoc) : null;
            }
            if (call.campaignId) {
                const campDoc = await db.collection('campaigns').doc(call.campaignId).get();
                call.campaign = campDoc.exists ? docToObj(campDoc) : null;
            }
        }

        res.json({ calls: paginated, total });
    } catch (error) {
        console.error("Error fetching calls:", error);
        res.status(500).json({ error: "Internal server error" });
    }
});

// GET /api/analytics/calls — List ALL calls for current user (direct userId + agentId fallback)
router.get('/calls', async (req, res) => {
    try {
        const { page = 1, limit = 50, sentiment, minDuration, disposition, status, startDate, endDate } = req.query;
        const pageNum = Math.max(1, parseInt(page, 10) || 1);
        const limitNum = Math.min(200, Math.max(1, parseInt(limit, 10) || 50));
        const userId = req.userId;

        console.log(`[ANALYTICS] GET /calls — userId: ${userId}`);

        // 1. Get calls by direct userId match
        const directSnap = await db.collection('calls').where('userId', '==', userId).get();
        const callsMap = new Map();
        directSnap.docs.forEach(doc => callsMap.set(doc.id, { id: doc.id, ...doc.data() }));
        console.log(`[ANALYTICS] Direct userId match: ${callsMap.size} calls`);

        // 2. Also get calls linked to this user's agents (handles older calls missing userId field)
        const agentsSnap = await db.collection('agents').where('userId', '==', userId).get();
        const userAgentIds = agentsSnap.docs.map(d => d.id);
        console.log(`[ANALYTICS] User owns ${userAgentIds.length} agents`);

        if (userAgentIds.length > 0) {
            for (let i = 0; i < userAgentIds.length; i += 30) {
                const chunk = userAgentIds.slice(i, i + 30);
                const agentCallsSnap = await db.collection('calls').where('agentId', 'in', chunk).get();
                agentCallsSnap.forEach(doc => {
                    if (!callsMap.has(doc.id)) callsMap.set(doc.id, { id: doc.id, ...doc.data() });
                });
            }
        }

        let calls = Array.from(callsMap.values());
        console.log(`[ANALYTICS] Total calls found: ${calls.length}`);

        // Apply filters
        if (startDate) {
            const startMs = new Date(startDate).getTime();
            if (!isNaN(startMs)) {
                calls = calls.filter(c => {
                    const ts = c.startedAt?.toDate ? c.startedAt.toDate().getTime() : new Date(c.startedAt || 0).getTime();
                    return ts >= startMs;
                });
            }
        }
        if (endDate) {
            let endObj = new Date(endDate);
            if (!isNaN(endObj.getTime())) {
                // If only a date was passed, bump it to the end of that day (23:59:59.999)
                if (endDate.trim().length <= 10) {
                    endObj.setUTCHours(23, 59, 59, 999);
                }
                const endMs = endObj.getTime();
                calls = calls.filter(c => {
                    const ts = c.startedAt?.toDate ? c.startedAt.toDate().getTime() : new Date(c.startedAt || 0).getTime();
                    return ts <= endMs;
                });
            }
        }
        if (status) calls = calls.filter(c => c.status === status);
        if (sentiment) calls = calls.filter(c => c.sentiment === sentiment);
        if (minDuration) calls = calls.filter(c => (c.duration || 0) >= parseInt(minDuration, 10));
        if (disposition) calls = calls.filter(c => c.dispositionId === disposition);

        // Sort by startedAt descending (newest first)
        calls.sort((a, b) => {
            const ta = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
            const tb = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
            return tb - ta;
        });

        const total = calls.length;
        const paginated = calls.slice((pageNum - 1) * limitNum, pageNum * limitNum);

        // Enrich with agent name and ensure consistent transcript structure
        for (const call of paginated) {
            if (call.agentId && !call.agentName) {
                try {
                    const agentDoc = await db.collection('agents').doc(call.agentId).get();
                    call.agentName = agentDoc.exists ? agentDoc.data().name : 'Unknown';
                } catch (e) { /* ignore */ }
            }
            // Ensure transcript fields are always present and properly formatted
            call.transcript = call.transcript || '';
            call.transcriptMessages = call.transcriptMessages || [];
            call.summary = call.summary || null;
            call.outcome = call.outcome || null;
            call.disposition = call.disposition || call.outcome || 'Unclear';
            call.sentiment = call.sentiment || 'neutral';
            call.notes = call.notes || null;
            call.agreed = call.agreed || false;
            call.commitmentDate = call.commitmentDate || null;
            try { call.structuredData = call.structuredData ? (typeof call.structuredData === 'string' ? JSON.parse(call.structuredData) : call.structuredData) : null; } catch(e) { call.structuredData = null; }
            call.recordingUrl = call.recordingUrl || call.recordingFilename || null;
        }

        res.json({ calls: paginated, total, pagination: { page: pageNum, limit: limitNum, totalPages: Math.ceil(total / limitNum) } });
    } catch (error) {
        console.error("[ANALYTICS] Error fetching call logs:", error);
        res.status(500).json({ error: "Failed to fetch call logs" });
    }
});

// GET /api/analytics/calls/:id
router.get('/calls/:id', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        // Allow access if userId matches OR if the call belongs to this user's agent
        if (call.userId && call.userId !== req.userId) {
            if (call.agentId) {
                const agentDoc = await db.collection('agents').doc(call.agentId).get();
                if (!agentDoc.exists || agentDoc.data().userId !== req.userId) {
                    return res.status(404).json({ error: 'Call not found' });
                }
            } else {
                return res.status(404).json({ error: 'Call not found' });
            }
        }

        if (call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            call.agent = agentDoc.exists ? { name: agentDoc.data().name } : null;
            if (!call.agentName && agentDoc.exists) call.agentName = agentDoc.data().name;
        }

        // Ensure all transcript fields are present
        call.transcript = call.transcript || '';
        call.transcriptMessages = call.transcriptMessages || [];
        call.summary = call.summary || null;
        call.outcome = call.outcome || null;
        call.disposition = call.disposition || call.outcome || 'Unclear';
        call.sentiment = call.sentiment || 'neutral';
        call.notes = call.notes || null;
        call.agreed = call.agreed || false;
        call.commitmentDate = call.commitmentDate || null;
        try { call.structuredData = call.structuredData ? (typeof call.structuredData === 'string' ? JSON.parse(call.structuredData) : call.structuredData) : null; } catch(e) { call.structuredData = null; }
        call.recordingUrl = call.recordingUrl || call.recordingFilename || null;

        res.json(call);
    } catch (e) {
        console.error('[ANALYTICS] Call detail error:', e);
        res.status(500).json({ error: 'Failed to retrieve call' });
    }
});

// POST /api/analytics/calls/:id/acw
router.post('/calls/:id/acw', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        const rawTranscript = call.transcript || 'No transcript available.';
        const redactedMsg = rawTranscript
            .replace(/\b\d{10,}\b/g, '[REDACTED PHONE]')
            .replace(/\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/g, '****-****-****-****');

        if (!process.env.GENARTML_SERVER_KEY && !process.env.CX_LLM_KEY || process.env.$(echo R0VNSU5JX0FQSV9LRVk= | base64 -d)) {
            const updateData = { summary: `System Auto-Summary: Call with ${call.phoneNumber} lasted ${call.duration}s.`, redactedTranscript: redactedMsg };
            await db.collection('calls').doc(req.params.id).update(updateData);
            return res.json({ ...call, ...updateData });
        }

        // Fetch analysisSchema if agent or campaign exists
        let customSchemaStr = '[]';
        if (call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            if (agentDoc.exists && agentDoc.data().analysisSchema) customSchemaStr = agentDoc.data().analysisSchema;
        } else if (call.campaignId) {
            const campaignDoc = await db.collection('campaigns').doc(call.campaignId).get();
            if (campaignDoc.exists && campaignDoc.data().analysisSchema) customSchemaStr = campaignDoc.data().analysisSchema;
        }

        let customSchema = [];
        try { customSchema = JSON.parse(customSchemaStr); } catch (e) {}

        const { getCXModelClient } = await import('../_rctx.js');
        const ai = await getCXModelClient(process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env.$(echo R0VNSU5JX0FQSV9LRVk= | base64 -d));
        
        let customPromptPart = '';
        if (customSchema.length > 0) {
            customPromptPart = customSchema.map((field, i) => `${i + 3}. "${field.name}": ${field.type} - ${field.description}`).join('\n');
        } else {
            customPromptPart = `3. "intent": string - The primary reason for the call.\n4. "agreed": boolean.\n5. "followUpRequired": boolean.`;
        }

        const prompt = `Analyze this call transcript between an AI agent and a customer.\nTranscript:\n---\n${redactedMsg}\n---\n\nReturn ONLY raw strict JSON containing:\n1. "summary": A concise 1-2 sentence executive summary.\n2. "disposition": A short 1-3 word classification of the call outcome.\n${customPromptPart}`;

        const response = await ai.models.generateContent({ model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(), contents: prompt, config: { systemInstruction: "You are an expert Q&A Call Center Analyst. You must return EXACTLY valid JSON, without any markdown formatting or code blocks.", temperature: 0.2 } });

        let jsonText = (response.text || "{}").replace(/```(?:json)?/gi, '').replace(/```/g, '').trim();
        let structured = {};
        try { 
            structured = JSON.parse(jsonText); 
        } catch { 
            structured = { summary: jsonText.substring(0, 200) }; 
        }

        const summaryText = structured.summary || 'Summary unavailable.';
        delete structured.summary; // Remove from structuredData so it doesn't double-render
        if (structured.disposition) delete structured.disposition; // Optional cleanup

        const updateData = {
            summary: summaryText,
            structuredData: JSON.stringify(structured),
            redactedTranscript: redactedMsg
        };
        await db.collection('calls').doc(req.params.id).update(updateData);
        res.json({ ...call, ...updateData });
    } catch (err) {
        console.error('[ACW Error]', err);
        res.status(500).json({ error: 'AI Summarization failed' });
    }
});

// GET /api/analytics/stats
router.get('/stats', async (req, res) => {
    const snap = await db.collection('calls').get();
    const calls = queryToArray(snap);
    const total = calls.length;
    const completed = calls.filter(c => c.status === 'completed').length;
    const sentiment = {
        positive: calls.filter(c => c.sentiment === 'positive').length,
        neutral: calls.filter(c => c.sentiment === 'neutral').length,
        negative: calls.filter(c => c.sentiment === 'negative').length,
        angry: calls.filter(c => c.sentiment === 'angry').length,
    };
    res.json({ total, completed, sentiment });
});

export default router;
