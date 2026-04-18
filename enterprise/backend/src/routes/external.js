import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import { requireApiKey } from '../middleware/auth.js';
import multer from 'multer';

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } }); // 20MB max
router.use(requireApiKey);

const parseBool = (val, def) => {
    if (val === undefined || val === null || val === '') return def;
    if (typeof val === 'string') return val.toLowerCase() === 'true' || val === '1';
    return Boolean(val);
};

const parseNum = (val, def) => {
    if (val === undefined || val === null || val === '') return def;
    const n = Number(val);
    return isNaN(n) ? def : n;
};

// GET /v1/agents
router.get('/agents', async (req, res) => {
    try {
        const page = Math.max(1, parseInt(req.query.page) || 1);
        const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 10));
        const status = req.query.status;

        let query = db.collection('agents').where('userId', '==', req.apiUser.userId);
        const snap = await query.get();
        let agents = queryToArray(snap);
        if (status) agents = agents.filter(a => a.status === status);

        // Sort by createdAt descending
        agents.sort((a, b) => {
            const ta = a.createdAt?.toDate ? a.createdAt.toDate().getTime() : new Date(a.createdAt || 0).getTime();
            const tb = b.createdAt?.toDate ? b.createdAt.toDate().getTime() : new Date(b.createdAt || 0).getTime();
            return tb - ta;
        });

        const total = agents.length;
        const paginated = agents.slice((page - 1) * limit, page * limit);

        res.json({ agents: paginated, pagination: { page, limit, total, totalPages: Math.ceil(total / limit) } });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to list agents' });
    }
});

// POST /v1/agents
router.post('/agents', upload.single('file'), async (req, res) => {
    try {
        const { name, description, systemPrompt, openingLine, voice, language, sttEngine, llmModel,
            fillerPhrases, prosodyRate, prosodyPitch, ipaLexicon, tools, topK, similarityThresh,
            fallbackMessage, profanityFilter, topicRestriction, backgroundAmbience, speakingStyle,
            bargeInMode, patienceMs, maxDuration, temperature, maxTokens, strictToolCalling,
            ringTimeout, voicemailLogic, webhookUrl, autoSummary, autoSentiment, recordCall, processDtmf,
            amdPrecision, voicemailDropAudio, sentimentRouting, competitorAlerts, supervisorWhisper,
            piiRedaction, geoCallerId, multiAgentHandoff, objectionHandling, emotionalMirroring,
            complianceScript, dynamicCodeSwitching, dncLitigatorScrub, callBlending, costCapTokens,
            postCallSms, autoFollowUp, followUpDefaultDays, followUpDefaultTime,
            dispositions, analysisSchema
        } = req.body;

        if (!name) return res.status(400).json({ error: "Agent 'name' is required." });

        const data = {
            userId: req.apiUser.userId,
            name, description: description || '', systemPrompt: systemPrompt || '', openingLine: openingLine || '',
            voice: voice || 'MF4J4IDTRo0AxOO4dpFR', language: language || 'en-US', sttEngine: sttEngine || 'callex-1.1',
            llmModel: llmModel || 'callex-1.3',
            fillerPhrases: typeof fillerPhrases === 'string' ? fillerPhrases : JSON.stringify(fillerPhrases || ['Let me check...', 'One moment...']),
            prosodyRate: parseNum(prosodyRate, 1.0), prosodyPitch: parseNum(prosodyPitch, 1.0),
            ipaLexicon: typeof ipaLexicon === 'string' ? ipaLexicon : JSON.stringify(ipaLexicon || {}),
            tools: typeof tools === 'string' ? tools : JSON.stringify(tools || []),
            topK: parseNum(topK, 5), similarityThresh: parseNum(similarityThresh, 0.75),
            fallbackMessage: fallbackMessage || "I'm sorry, I don't have that information right now.",
            profanityFilter: profanityFilter || 'redact',
            topicRestriction: parseBool(topicRestriction, false),
            backgroundAmbience: backgroundAmbience || 'none',
            speakingStyle: speakingStyle || 'professional',
            bargeInMode: bargeInMode || 'balanced',
            patienceMs: parseNum(patienceMs, 800),
            maxDuration: parseNum(maxDuration, 30),
            temperature: parseNum(temperature, 0.7),
            maxTokens: parseNum(maxTokens, 250),
            strictToolCalling: parseBool(strictToolCalling, true),
            ringTimeout: parseNum(ringTimeout, 30),
            voicemailLogic: voicemailLogic || 'hangup',
            webhookUrl: webhookUrl || null,
            autoSummary: parseBool(autoSummary, true),
            autoSentiment: parseBool(autoSentiment, true),
            recordCall: parseBool(recordCall, true),
            processDtmf: parseBool(processDtmf, true),
            amdPrecision: amdPrecision || 'balanced',
            voicemailDropAudio: voicemailDropAudio || null,
            sentimentRouting: parseBool(sentimentRouting, false),
            competitorAlerts: competitorAlerts || '',
            supervisorWhisper: parseBool(supervisorWhisper, true),
            piiRedaction: parseBool(piiRedaction, true),
            geoCallerId: parseBool(geoCallerId, false),
            multiAgentHandoff: parseBool(multiAgentHandoff, false),
            objectionHandling: objectionHandling || 'standard',
            emotionalMirroring: parseBool(emotionalMirroring, true),
            complianceScript: complianceScript || null,
            dynamicCodeSwitching: parseBool(dynamicCodeSwitching, true),
            dncLitigatorScrub: parseBool(dncLitigatorScrub, true),
            callBlending: parseBool(callBlending, false),
            costCapTokens: parseNum(costCapTokens, 5000),
            postCallSms: postCallSms || null,
            autoFollowUp: parseBool(autoFollowUp, true),
            followUpDefaultDays: parseNum(followUpDefaultDays, 1),
            followUpDefaultTime: followUpDefaultTime || '10:00',
            analysisSchema: typeof analysisSchema === 'string' ? analysisSchema : JSON.stringify(analysisSchema || []),
            status: 'draft',
            createdAt: new Date(),
            updatedAt: new Date(),
        };

        // If file is uploaded during creation, process knowledge immediately
        if (req.file) {
            const { buffer, mimetype, originalname } = req.file;
            const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env[Buffer.from('R0VNSU5JX0FQSV9LRVk=', 'base64').toString()];
            
            if (CX_LLM_KEY) {
                let knowledgeText = '';
                let rawText = await extractFileText(buffer, mimetype, originalname);

                if (rawText) {
                    const { getCXModelClient } = await import('../_rctx.js');
                    const genAI = await getCXModelClient(CX_LLM_KEY);
                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{
                            role: 'user',
                            parts: [{ text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information from this text content.\n\nOutput a clean, structured knowledge base in this EXACT format:\nKNOWLEDGE BASE:\n[Write all extracted information as clear Q&A pairs, facts, pricing details, product info, policies, etc.]\n\nTOPICS COVERED:\n[Comma-separated list of main topics found]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[List 5 example customer questions]\n\nHere is the content:\n${rawText}` }]
                        }]
                    });
                    knowledgeText = response.text;
                } else {
                    knowledgeText = await parseDocumentWithAI(buffer, mimetype, originalname, CX_LLM_KEY);
                }

                if (knowledgeText && knowledgeText.length > 50) {
                    const topicsMatch = knowledgeText.match(/TOPICS COVERED:\s*\n?(.*?)(?:\n\n|\nTOTAL|\nSAMPLE)/s);
                    const totalMatch = knowledgeText.match(/TOTAL ITEMS:\s*\n?(\d+)/);
                    const sampleMatch = knowledgeText.match(/SAMPLE QUESTIONS:\s*\n?([\s\S]*?)$/);
                    const knowledgeMatch = knowledgeText.match(/KNOWLEDGE BASE:\s*\n?([\s\S]*?)(?:\nTOPICS COVERED)/);

                    data.knowledgeTopics = topicsMatch ? topicsMatch[1].trim().split(',').map(t => t.trim()).filter(Boolean) : [];
                    data.knowledgeBase = knowledgeMatch ? knowledgeMatch[1].trim() : knowledgeText;
                    
                    data.trainingSummary = {
                        agentName: data.name,
                        purpose: data.description || 'General purpose calling agent',
                        openingLine: data.openingLine || '',
                        knowledgeTopics: data.knowledgeTopics,
                        totalFaqs: totalMatch ? parseInt(totalMatch[1]) : 0,
                        sampleQuestions: sampleMatch ? sampleMatch[1].trim().split('\n').map(q => q.replace(/^[-\d.)\s]+/, '').trim()).filter(q => q.length > 5).slice(0, 5) : [],
                        lastTrainedAt: new Date().toISOString(),
                        lastTrainedFile: originalname,
                        hasSystemPrompt: !!(data.systemPrompt && data.systemPrompt.length > 10),
                        hasKnowledgeBase: true,
                    };
                }
            }
        }

        const ref = await db.collection('agents').add(data);
        const agent = { id: ref.id, ...data };

        await db.collection('promptVersions').add({ agentId: ref.id, version: 1, prompt: systemPrompt || '', isActive: true, label: 'v1 - Initial', createdAt: new Date() });

        // Handle optional unified dispositions
        let createdDispositions = [];
        if (dispositions) {
            try {
                const parsedDispositions = typeof dispositions === 'string' ? JSON.parse(dispositions) : dispositions;
                if (Array.isArray(parsedDispositions)) {
                    for (let d of parsedDispositions) {
                        if (!d.name) continue;
                        const dispData = {
                            name: d.name,
                            category: d.category || 'General',
                            requiresNote: d.requiresNote || false,
                            active: true,
                            userId: req.apiUser.userId,
                            createdAt: new Date()
                        };
                        const dispRef = await db.collection('dispositions').add(dispData);
                        createdDispositions.push({ id: dispRef.id, ...dispData });
                    }
                }
            } catch (err) {
                console.error('[AGENT CREATION] Failed to parse dispositions:', err);
            }
        }
        
        if (createdDispositions.length > 0) agent.createdDispositions = createdDispositions;

        res.status(201).json({ message: "Agent successfully created.", agentId: ref.id, agent });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: "Failed to create agent" });
    }
});

// GET /v1/agents/:id
router.get('/agents/:id', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const agent = docToObj(doc);
        if (!agent || agent.userId !== req.apiUser.userId) return res.status(404).json({ error: 'Agent not found' });
        const pvSnap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
        const versions = queryToArray(pvSnap);
        versions.sort((a, b) => (b.version || 0) - (a.version || 0));
        agent.PromptVersion = versions;
        res.json(agent);
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: "Failed to retrieve agent" });
    }
});

// PUT /v1/agents/:id
router.put('/agents/:id', upload.single('file'), async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.apiUser.userId) return res.status(404).json({ error: 'Agent not found' });

        const updates = { ...req.body, updatedAt: new Date() };

        // JSON Stringified Arrays
        if (updates.fillerPhrases !== undefined) updates.fillerPhrases = typeof updates.fillerPhrases === 'string' ? updates.fillerPhrases : JSON.stringify(updates.fillerPhrases);
        if (updates.ipaLexicon !== undefined) updates.ipaLexicon = typeof updates.ipaLexicon === 'string' ? updates.ipaLexicon : JSON.stringify(updates.ipaLexicon);
        if (updates.tools !== undefined) updates.tools = typeof updates.tools === 'string' ? updates.tools : JSON.stringify(updates.tools);
        if (updates.analysisSchema !== undefined) updates.analysisSchema = typeof updates.analysisSchema === 'string' ? updates.analysisSchema : JSON.stringify(updates.analysisSchema);

        // Booleans
        const boolFields = ['topicRestriction', 'strictToolCalling', 'autoSummary', 'autoSentiment', 'recordCall', 'processDtmf', 'sentimentRouting', 'supervisorWhisper', 'piiRedaction', 'geoCallerId', 'multiAgentHandoff', 'emotionalMirroring', 'dynamicCodeSwitching', 'dncLitigatorScrub', 'callBlending', 'autoFollowUp'];
        for (const f of boolFields) {
            if (updates[f] !== undefined) updates[f] = parseBool(updates[f], updates[f]);
        }

        // Numbers
        const numFields = ['prosodyRate', 'prosodyPitch', 'topK', 'similarityThresh', 'patienceMs', 'maxDuration', 'temperature', 'maxTokens', 'ringTimeout', 'costCapTokens', 'followUpDefaultDays'];
        for (const f of numFields) {
            if (updates[f] !== undefined) updates[f] = parseNum(updates[f], updates[f]);
        }

        // Handle Knowledge Base Upload if file is present
        if (req.file) {
            const { buffer, mimetype, originalname } = req.file;
            const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env[Buffer.from('R0VNSU5JX0FQSV9LRVk=', 'base64').toString()];
            
            if (CX_LLM_KEY) {
                let knowledgeText = '';
                let rawText = await extractFileText(buffer, mimetype, originalname);

                if (rawText) {
                    const { getCXModelClient } = await import('../_rctx.js');
                    const genAI = await getCXModelClient(CX_LLM_KEY);
                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{
                            role: 'user',
                            parts: [{ text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information from this text content.\n\nOutput a clean, structured knowledge base in this EXACT format:\nKNOWLEDGE BASE:\n[Write all extracted information as clear Q&A pairs, facts, pricing details, product info, policies, etc.]\n\nTOPICS COVERED:\n[Comma-separated list of main topics found]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[List 5 example customer questions]\n\nHere is the content:\n${rawText}` }]
                        }]
                    });
                    knowledgeText = response.text;
                } else {
                    knowledgeText = await parseDocumentWithAI(buffer, mimetype, originalname, CX_LLM_KEY);
                }

                if (knowledgeText && knowledgeText.length > 50) {
                    const topicsMatch = knowledgeText.match(/TOPICS COVERED:\s*\n?(.*?)(?:\n\n|\nTOTAL|\nSAMPLE)/s);
                    const totalMatch = knowledgeText.match(/TOTAL ITEMS:\s*\n?(\d+)/);
                    const sampleMatch = knowledgeText.match(/SAMPLE QUESTIONS:\s*\n?([\s\S]*?)$/);
                    const knowledgeMatch = knowledgeText.match(/KNOWLEDGE BASE:\s*\n?([\s\S]*?)(?:\nTOPICS COVERED)/);

                    updates.knowledgeTopics = topicsMatch ? topicsMatch[1].trim().split(',').map(t => t.trim()).filter(Boolean) : [];
                    updates.knowledgeBase = knowledgeMatch ? knowledgeMatch[1].trim() : knowledgeText;
                    
                    updates.trainingSummary = {
                        agentName: existing.name || '',
                        purpose: existing.description || 'General purpose calling agent',
                        openingLine: existing.openingLine || '',
                        knowledgeTopics: updates.knowledgeTopics,
                        totalFaqs: totalMatch ? parseInt(totalMatch[1]) : 0,
                        sampleQuestions: sampleMatch ? sampleMatch[1].trim().split('\n').map(q => q.replace(/^[-\d.)\s]+/, '').trim()).filter(q => q.length > 5).slice(0, 5) : [],
                        lastTrainedAt: new Date().toISOString(),
                        lastTrainedFile: originalname,
                        hasSystemPrompt: existing.systemPrompt ? true : false,
                        hasKnowledgeBase: true,
                    };
                }
            }
        }

        // Handle system prompt clear or update dynamically outside of the prompt tab
        if (updates.systemPrompt !== undefined) {
            try {
                // Fetch ALL prompt versions (no orderBy = no composite index needed)
                const allPvSnap = await db.collection('promptVersions')
                    .where('agentId', '==', req.params.id).get();
                
                let nextVersion = 1;
                if (!allPvSnap.empty) {
                    let maxVersion = 0;
                    for (const pvDoc of allPvSnap.docs) {
                        const v = pvDoc.data().version || 0;
                        if (v > maxVersion) maxVersion = v;
                        await db.collection('promptVersions').doc(pvDoc.id).update({ isActive: false });
                    }
                    nextVersion = maxVersion + 1;
                }
                
                await db.collection('promptVersions').add({
                    agentId: req.params.id,
                    version: nextVersion,
                    prompt: updates.systemPrompt,
                    isActive: true,
                    label: `v${nextVersion} - Edit Settings Update`,
                    createdAt: new Date()
                });
                console.log(`[EXTERNAL API] ✅ Prompt version v${nextVersion} saved for agent ${req.params.id}`);
            } catch(err) {
                console.error('[AGENT EDIT] Failed to save prompt version:', err);
            }
        }

        delete updates.id; delete updates.createdAt;

        await db.collection('agents').doc(req.params.id).update(updates);
        const updated = await db.collection('agents').doc(req.params.id).get();
        res.json({ message: 'Agent updated successfully.', agentId: req.params.id, agent: docToObj(updated) });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to update agent' });
    }
});
// PATCH /v1/agents/:id/prompt
router.patch('/agents/:id/prompt', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.apiUser.userId) {
            return res.status(404).json({ error: 'Agent not found' });
        }

        const { systemPrompt } = req.body;
        if (typeof systemPrompt !== 'string') {
            return res.status(400).json({ error: 'systemPrompt must be a string' });
        }

        // Update the main agent document directly
        await db.collection('agents').doc(req.params.id).update({ 
            systemPrompt: systemPrompt,
            updatedAt: new Date()
        });

        // Track version history
        try {
            // Fetch ALL prompt versions (no orderBy = no composite index needed)
            const allPvSnap = await db.collection('promptVersions')
                .where('agentId', '==', req.params.id).get();
            
            let nextVersion = 1;
            if (!allPvSnap.empty) {
                let maxVersion = 0;
                for (const pvDoc of allPvSnap.docs) {
                    const v = pvDoc.data().version || 0;
                    if (v > maxVersion) maxVersion = v;
                    await db.collection('promptVersions').doc(pvDoc.id).update({ isActive: false });
                }
                nextVersion = maxVersion + 1;
            }
            
            await db.collection('promptVersions').add({
                agentId: req.params.id,
                version: nextVersion,
                prompt: systemPrompt,
                isActive: true,
                label: `v${nextVersion} - API Update`,
                createdAt: new Date()
            });
            console.log(`[EXTERNAL API] ✅ Prompt version v${nextVersion} saved for agent ${req.params.id}`);
        } catch(err) {
            console.error('[AGENT EDIT] Failed to save prompt version:', err);
        }

        const updated = await db.collection('agents').doc(req.params.id).get();
        res.json({ message: 'System prompt updated successfully.', agentId: req.params.id, agent: docToObj(updated) });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to update system prompt' });
    }
});

// DELETE /v1/agents/:id
router.delete('/agents/:id', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.apiUser.userId) return res.status(404).json({ error: 'Agent not found' });

        // Delete related records
        const pvSnap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
        const fuSnap = await db.collection('followUps').where('agentId', '==', req.params.id).get();
        const batch = db.batch();
        pvSnap.forEach(d => batch.delete(d.ref));
        fuSnap.forEach(d => batch.delete(d.ref));
        batch.delete(db.collection('agents').doc(req.params.id));
        await batch.commit();

        res.json({ message: 'Agent deleted successfully.', agentId: req.params.id });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to delete agent' });
    }
});

// ═══════════════════════════════════════════════
// KNOWLEDGE UPLOAD (Agent Training from PDF/Excel/CSV/TXT)
// ═══════════════════════════════════════════════

/**
 * Helper: Extract raw text from uploaded file buffer based on MIME type.
 * Supports PDF (via AI vision), Excel, CSV, and plain text.
 */
async function extractFileText(fileBuffer, mimetype, originalname) {
    const textContent = [];

    // ── Plain Text / CSV ──
    if (mimetype === 'text/plain' || mimetype === 'text/csv' || originalname.endsWith('.csv') || originalname.endsWith('.txt')) {
        return fileBuffer.toString('utf-8');
    }

    // ── Excel (.xlsx, .xls) ──
    if (mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
        mimetype === 'application/vnd.ms-excel' ||
        originalname.endsWith('.xlsx') || originalname.endsWith('.xls')) {
        // Parse Excel using a simple row-by-row extraction
        // We'll send the base64 to AI for intelligent parsing
        return null; // Signal to use AI vision for Excel too
    }

    // ── PDF ──
    if (mimetype === 'application/pdf' || originalname.endsWith('.pdf')) {
        return null; // Signal to use AI vision
    }

    // ── Unsupported ──
    return null;
}

/**
 * Use AI to intelligently parse a document (PDF/Excel/image) into structured knowledge.
 */
async function parseDocumentWithAI(fileBuffer, mimetype, originalname, apiKey) {
    const { getCXModelClient } = await import('../_rctx.js');
    const genAI = await getCXModelClient(apiKey);

    const base64Data = fileBuffer.toString('base64');

    // Map common mimetypes for AI document parser
    let docMimeType = mimetype;
    if (originalname.endsWith('.xlsx')) docMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    if (originalname.endsWith('.xls')) docMimeType = 'application/vnd.ms-excel';
    if (originalname.endsWith('.csv')) docMimeType = 'text/csv';

    const response = await genAI.models.generateContent({
        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
        contents: [{
            role: 'user',
            parts: [
                {
                    inlineData: {
                        mimeType: docMimeType,
                        data: base64Data
                    }
                },
                {
                    text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information from this document.

Output a clean, structured knowledge base in this EXACT format:

KNOWLEDGE BASE:
[Write all extracted information as clear Q&A pairs, facts, pricing details, product info, policies, etc. Organize by topic. Use simple language that a phone agent can speak naturally.]

TOPICS COVERED:
[Comma-separated list of main topics found]

TOTAL ITEMS:
[Number of distinct knowledge items/FAQs extracted]

SAMPLE QUESTIONS:
[List 5 example questions a customer might ask that this knowledge can answer]

Rules:
- Extract EVERY piece of information, don't skip anything
- Convert tables/charts into readable text
- Convert pricing into spoken format (e.g., "twenty five lakh rupees" not "₹25L")
- If in Hindi, keep it in Hindi. If English, keep English. If mixed, keep mixed.
- Be thorough — this knowledge will be the agent's entire brain for calls`
                }
            ]
        }]
    });

    return response.text;
}

// POST /v1/agents/:id/knowledge — Upload document to train agent
router.post('/agents/:id/knowledge', upload.single('file'), async (req, res) => {
    try {
        // 1. Verify agent ownership
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.apiUser.userId) {
            return res.status(404).json({ error: 'Agent not found' });
        }

        // 2. Validate file
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded. Send a file with field name "file".' });
        }

        const { buffer, mimetype, originalname, size } = req.file;
        const allowedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'text/csv', 'text/plain',
        ];
        const allowedExtensions = ['.pdf', '.xlsx', '.xls', '.csv', '.txt'];
        const ext = '.' + originalname.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(mimetype) && !allowedExtensions.includes(ext)) {
            return res.status(400).json({
                error: 'Unsupported file type. Allowed: PDF, Excel (.xlsx/.xls), CSV, TXT',
                received: { mimetype, extension: ext }
            });
        }

        console.log(`[KNOWLEDGE] Processing ${originalname} (${(size / 1024).toFixed(1)}KB) for agent ${req.params.id}`);

        // 3. Extract text content
        const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env[Buffer.from('R0VNSU5JX0FQSV9LRVk=', 'base64').toString()];
        if (!CX_LLM_KEY) {
            return res.status(500).json({ error: 'Server configuration error: AI API key not set' });
        }

        let knowledgeText = '';
        let rawText = await extractFileText(buffer, mimetype, originalname);

        if (rawText) {
            // For plain text/CSV, we still send to AI for intelligent structuring
            const { getCXModelClient } = await import('../_rctx.js');
            const genAI = await getCXModelClient(CX_LLM_KEY);

            const response = await genAI.models.generateContent({
                model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                contents: [{
                    role: 'user',
                    parts: [{
                        text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information from this text content.

Output a clean, structured knowledge base in this EXACT format:

KNOWLEDGE BASE:
[Write all extracted information as clear Q&A pairs, facts, pricing details, product info, policies, etc.]

TOPICS COVERED:
[Comma-separated list of main topics found]

TOTAL ITEMS:
[Number of distinct knowledge items/FAQs extracted]

SAMPLE QUESTIONS:
[List 5 example questions a customer might ask that this knowledge can answer]

Here is the content:
${rawText}`
                    }]
                }]
            });
            knowledgeText = response.text;
        } else {
            // For PDF/Excel, use AI vision to parse the file directly
            knowledgeText = await parseDocumentWithAI(buffer, mimetype, originalname, CX_LLM_KEY);
        }

        if (!knowledgeText || knowledgeText.length < 50) {
            return res.status(422).json({ error: 'Could not extract meaningful knowledge from the uploaded file. Please try a different file.' });
        }

        // 4. Parse the structured output to extract metadata
        const topicsMatch = knowledgeText.match(/TOPICS COVERED:\s*\n?(.*?)(?:\n\n|\nTOTAL|\nSAMPLE)/s);
        const totalMatch = knowledgeText.match(/TOTAL ITEMS:\s*\n?(\d+)/);
        const sampleMatch = knowledgeText.match(/SAMPLE QUESTIONS:\s*\n?([\s\S]*?)$/);
        const knowledgeMatch = knowledgeText.match(/KNOWLEDGE BASE:\s*\n?([\s\S]*?)(?:\nTOPICS COVERED)/);

        const topics = topicsMatch ? topicsMatch[1].trim().split(',').map(t => t.trim()).filter(Boolean) : [];
        const totalItems = totalMatch ? parseInt(totalMatch[1]) : 0;
        const sampleQuestions = sampleMatch
            ? sampleMatch[1].trim().split('\n').map(q => q.replace(/^[-\d.)\s]+/, '').trim()).filter(q => q.length > 5).slice(0, 5)
            : [];
        const extractedKnowledge = knowledgeMatch ? knowledgeMatch[1].trim() : knowledgeText;

        // 5. Merge with existing knowledge (append, don't replace)
        const existingKnowledge = existing.knowledgeBase || '';
        const mergedKnowledge = existingKnowledge
            ? `${existingKnowledge}\n\n--- New Knowledge (from ${originalname}) ---\n\n${extractedKnowledge}`
            : extractedKnowledge;

        // 6. Generate training summary
        const trainingSummary = {
            agentName: existing.name,
            purpose: existing.description || 'General purpose calling agent',
            openingLine: existing.openingLine || '',
            knowledgeTopics: [...new Set([...(existing.knowledgeTopics || []), ...topics])],
            totalFaqs: totalItems,
            sampleQuestions,
            lastTrainedAt: new Date().toISOString(),
            lastTrainedFile: originalname,
            hasSystemPrompt: !!(existing.systemPrompt && existing.systemPrompt.length > 10),
            hasKnowledgeBase: true,
        };

        // 7. Save to Firestore
        await db.collection('agents').doc(req.params.id).update({
            knowledgeBase: mergedKnowledge,
            knowledgeTopics: trainingSummary.knowledgeTopics,
            trainingSummary,
            updatedAt: new Date(),
        });

        console.log(`[KNOWLEDGE] ✅ Agent ${req.params.id} trained with ${originalname} (${totalItems} items, ${topics.length} topics)`);

        res.json({
            message: 'Knowledge uploaded and processed successfully',
            trainingSummary,
            knowledgeSize: mergedKnowledge.length,
        });
    } catch (e) {
        console.error('[KNOWLEDGE ERROR]', e);
        res.status(500).json({ error: 'Failed to process knowledge file', details: e.message });
    }
});

// DELETE /v1/agents/:id/knowledge — Clear agent knowledge base
router.delete('/agents/:id/knowledge', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.apiUser.userId) {
            return res.status(404).json({ error: 'Agent not found' });
        }

        await db.collection('agents').doc(req.params.id).update({
            knowledgeBase: '',
            knowledgeTopics: [],
            trainingSummary: null,
            updatedAt: new Date(),
        });

        res.json({ message: 'Knowledge base cleared successfully' });
    } catch (e) {
        console.error('[KNOWLEDGE ERROR]', e);
        res.status(500).json({ error: 'Failed to clear knowledge base' });
    }
});

// ═══════════════════════════════════════════════
// CALLS API
// ═══════════════════════════════════════════════

// GET /v1/calls — List calls (paginated)
router.get('/calls', async (req, res) => {
    try {
        const page = Math.max(1, parseInt(req.query.page) || 1);
        const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));
        const { status, agentId, startDate, endDate } = req.query;
        const apiUserId = req.apiUser.userId;
        const isSuperAdmin = apiUserId === 'superadmin-hardcoded-id';

        console.log(`[EXT-API] GET /v1/calls — apiUserId: ${apiUserId}, superAdmin: ${isSuperAdmin}, status: ${status || 'any'}, agentId: ${agentId || 'any'}`);

        let callsMap = new Map();

        if (isSuperAdmin) {
            const allSnap = await db.collection('calls').get();
            allSnap.docs.forEach(doc => callsMap.set(doc.id, { id: doc.id, ...doc.data() }));
        } else {
            // 1. Get calls directly owned by this user
            const directSnap = await db.collection('calls').where('userId', '==', apiUserId).get();
            directSnap.docs.forEach(doc => {
                callsMap.set(doc.id, { id: doc.id, ...doc.data() });
            });
            console.log(`[EXT-API] Direct userId match: ${callsMap.size} calls`);

            // 2. Fallback: also get calls that belong to this user's agents but lack userId
            const agentsSnap = await db.collection('agents').where('userId', '==', apiUserId).get();
            const userAgentIds = agentsSnap.docs.map(d => d.id);
            console.log(`[EXT-API] User owns ${userAgentIds.length} agents`);

            if (userAgentIds.length > 0) {
                // Query in chunks of 30 (Firestore 'in' limit)
                for (let i = 0; i < userAgentIds.length; i += 30) {
                    const chunk = userAgentIds.slice(i, i + 30);
                    const agentCallsSnap = await db.collection('calls').where('agentId', 'in', chunk).get();
                    agentCallsSnap.forEach(doc => {
                        if (!callsMap.has(doc.id)) {
                            callsMap.set(doc.id, { id: doc.id, ...doc.data() });
                        }
                    });
                }
            }
        }

        let calls = Array.from(callsMap.values());
        console.log(`[EXT-API] Total calls found (direct + fallback): ${calls.length}`);

        // 3. Filter by status/agentId/date if provided (filtered in memory to avoid complex indexes)
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
        if (agentId) calls = calls.filter(c => c.agentId === agentId);

        // Sort by startedAt descending
        calls.sort((a, b) => {
            const ta = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
            const tb = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
            return tb - ta;
        });

        const total = calls.length;
        const paginated = calls.slice((page - 1) * limit, page * limit).map(c => ({
            id: c.id,
            phoneNumber: c.phoneNumber || '',
            crmId: c.crmId || null,
            agentId: c.agentId || '',
            agentName: c.agentName || '',
            status: c.status || 'unknown',
            duration: c.duration || 0,
            sentiment: c.sentiment || 'neutral',
            transcript: c.transcript || '',
            transcriptMessages: c.transcriptMessages || [],
            hasTranscript: !!(c.transcript && c.transcript.length > 0),
            hasRecording: !!(c.recordingUrl || c.recordingFilename),
            recordingUrl: c.recordingUrl || c.recordingFilename || null,
            summary: c.summary || null,
            outcome: c.outcome || null,
            notes: c.notes || null,
            agreed: c.agreed || false,
            commitmentDate: c.commitmentDate || null,
            disposition: c.disposition || c.outcome || 'Unclear',
            structuredData: c.structuredData ? (typeof c.structuredData === 'string' ? JSON.parse(c.structuredData) : c.structuredData) : null,
            startedAt: c.startedAt,
            endedAt: c.endedAt || null,
        }));

        console.log(`[EXT-API] Returning ${paginated.length} calls (page ${page}, total ${total})`);
        res.json({ calls: paginated, pagination: { page, limit, total, totalPages: Math.ceil(total / limit) } });
    } catch (e) {
        console.error('[EXT-API ERROR] GET /v1/calls failed:', e);
        res.status(500).json({ error: 'Failed to list calls' });
    }
});

// GET /v1/calls/:id — Get full call details including transcript
router.get('/calls/:id', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        const userId = req.apiUser.userId;
        const isSuperAdmin = userId === 'superadmin-hardcoded-id';
        let owned = isSuperAdmin || call.userId === userId;
        if (!owned && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            owned = agentDoc.exists && agentDoc.data().userId === userId;
        }
        if (!owned) return res.status(404).json({ error: 'Call not found' });

        // Get agent name if available
        let agentName = call.agentName || '';
        if (!agentName && call.agentId) {
            try {
                const agentDoc = await db.collection('agents').doc(call.agentId).get();
                if (agentDoc.exists) agentName = agentDoc.data().name || '';
            } catch (e) { /* ignore */ }
        }

        res.json({
            id: call.id,
            phoneNumber: call.phoneNumber || '',
            crmId: call.crmId || null,
            agentId: call.agentId || '',
            agentName,
            status: call.status || 'unknown',
            duration: call.duration || 0,
            sentiment: call.sentiment || 'neutral',
            transcript: call.transcript || '',
            transcriptMessages: call.transcriptMessages || [],
            recordingUrl: call.recordingUrl || call.recordingFilename || null,
            summary: call.summary || null,
            outcome: call.outcome || null,
            notes: call.notes || null,
            agreed: call.agreed || false,
            commitmentDate: call.commitmentDate || null,
            disposition: call.disposition || call.outcome || 'Unclear',
            structuredData: call.structuredData ? (typeof call.structuredData === 'string' ? JSON.parse(call.structuredData) : call.structuredData) : null,
            startedAt: call.startedAt,
            endedAt: call.endedAt || null,
        });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to retrieve call' });
    }
});

// PATCH /v1/calls/:id/disposition — Update the disposition for a specific call
router.patch('/calls/:id/disposition', async (req, res) => {
    try {
        const { id } = req.params;
        const { disposition, dispositionId } = req.body;
        const apiUserId = req.apiUser.userId;

        if (!disposition && !dispositionId) {
            return res.status(400).json({ error: "Missing 'disposition' or 'dispositionId' in request body" });
        }

        const callRef = db.collection('calls').doc(id);
        const doc = await callRef.get();

        if (!doc.exists) return res.status(404).json({ error: 'Call not found' });

        const call = doc.data();
        
        // Ownership check: allow if call belongs to apiUserId OR if call's agent belongs to apiUserId
        let isOwner = call.userId === apiUserId;
        if (!isOwner && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            if (agentDoc.exists && agentDoc.data().userId === apiUserId) {
                isOwner = true;
            }
        }

        if (!isOwner) return res.status(403).json({ error: 'Access denied to this call' });

        const updates = { updatedAt: new Date() };
        if (disposition !== undefined) updates.disposition = disposition;
        if (dispositionId !== undefined) updates.dispositionId = dispositionId;

        await callRef.update(updates);
        res.json({ message: 'Disposition updated successfully', id, ...updates });
    } catch (e) {
        console.error('[EXT-API ERROR] PATCH /v1/calls/:id/disposition:', e);
        res.status(500).json({ error: 'Failed to update call disposition' });
    }
});

// GET /v1/calls/:id/transcript — Get just the transcript for a call
router.get('/calls/:id/transcript', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        // Ownership check
        const userId = req.apiUser.userId;
        const isSuperAdmin = userId === 'superadmin-hardcoded-id';
        let owned = isSuperAdmin || call.userId === userId;
        if (!owned && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            owned = agentDoc.exists && agentDoc.data().userId === userId;
        }
        if (!owned) return res.status(404).json({ error: 'Call not found' });

        const messages = call.transcriptMessages || [];

        res.json({
            callId: call.id,
            phoneNumber: call.phoneNumber || '',
            crmId: call.crmId || null,
            agentId: call.agentId || '',
            agentName: call.agentName || '',
            duration: call.duration || 0,
            transcript: call.transcript || '',
            messages: messages,
            messageCount: messages.length,
            startedAt: call.startedAt,
            endedAt: call.endedAt || null,
        });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to retrieve transcript' });
    }
});

// GET /v1/calls/:id/transcript/live — Live polling endpoint for real-time transcription
// Query Params: ?since=<unix_timestamp> — only returns messages after this timestamp
router.get('/calls/:id/transcript/live', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        // Ownership check
        const userId = req.apiUser.userId;
        const isSuperAdmin = userId === 'superadmin-hardcoded-id';
        let owned = isSuperAdmin || call.userId === userId;
        if (!owned && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            owned = agentDoc.exists && agentDoc.data().userId === userId;
        }
        if (!owned) return res.status(404).json({ error: 'Call not found' });

        const allMessages = call.transcriptMessages || [];
        const since = parseFloat(req.query.since) || 0;

        // Filter messages that arrived AFTER the `since` timestamp
        const newMessages = since > 0
            ? allMessages.filter(m => (m.timestamp || 0) > since)
            : allMessages;

        // Determine if the call is still active (no endedAt means still live)
        const isLive = !call.endedAt;

        // Latest timestamp for the client to use as `since` in the next poll
        const latestTimestamp = newMessages.length > 0
            ? Math.max(...newMessages.map(m => m.timestamp || 0))
            : since;

        res.json({
            callId: call.id,
            isLive,
            messages: newMessages,
            newCount: newMessages.length,
            totalCount: allMessages.length,
            latestTimestamp,
            pollAgainAt: isLive ? latestTimestamp : null,
        });
    } catch (e) {
        console.error('[EXTERNAL API ERROR] GET /v1/calls/:id/transcript/live:', e);
        res.status(500).json({ error: 'Failed to retrieve live transcript' });
    }
});


// ═══════════════════════════════════════════════
// VOICES API
// ═══════════════════════════════════════════════

// GET /v1/voices — List all available Callex voices
router.get('/voices', async (req, res) => {
    try {
        const voices = [
            {
                id: 'MF4J4IDTRo0AxOO4dpFR',
                name: 'Devi',
                description: 'Clear Hindi female voice — crisp and natural',
                language: 'hi-IN',
                gender: 'female',
                style: 'professional',
                isDefault: true,
            },
            {
                id: '1qEiC6qsybMkmnNdVMbK',
                name: 'Monika',
                description: 'Modulated professional female voice',
                language: 'hi-IN',
                gender: 'female',
                style: 'professional',
                isDefault: false,
            },
            {
                id: 'qDuRKMlYmrm8trt5QyBn',
                name: 'Taksh',
                description: 'Powerful and commanding male voice',
                language: 'hi-IN',
                gender: 'male',
                style: 'authoritative',
                isDefault: false,
            },
            {
                id: 'LQ2auZHpAQ9h4azztqMT',
                name: 'Parveen',
                description: 'Confident male voice — warm and persuasive',
                language: 'hi-IN',
                gender: 'male',
                style: 'confident',
                isDefault: false,
            },
            {
                id: 's6cZdgI3j07hf4frz4Q8',
                name: 'Arvi',
                description: 'Desi conversational female voice — friendly and casual',
                language: 'hi-IN',
                gender: 'female',
                style: 'conversational',
                isDefault: false,
            },
        ];

        res.json({ voices, total: voices.length });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to list voices' });
    }
});

// ═══════════════════════════════════════════════
// SUPERVISOR API (LIVE CALLS)
// ═══════════════════════════════════════════════

// GET /v1/supervisor/calls — List all active calls
router.get('/supervisor/calls', async (req, res) => {
    try {
        const apiUserId = req.apiUser.userId;
        const isSuperAdmin = apiUserId === 'superadmin-hardcoded-id';

        // Query BOTH 'active' and 'in-progress' for maximum compatibility
        const [activeSnap, inProgressSnap] = await Promise.all([
            db.collection('calls').where('status', '==', 'active').get(),
            db.collection('calls').where('status', '==', 'in-progress').get(),
        ]);
        const combinedDocs = [...activeSnap.docs, ...inProgressSnap.docs];

        // Get this user's agent IDs for fallback matching
        const agentsSnap = await db.collection('agents').where('userId', '==', apiUserId).get();
        const userAgentIds = new Set(agentsSnap.docs.map(d => d.id));

        const calls = [];
        const ghostCleanupBatch = db.batch();
        let ghostCount = 0;
        const now = Date.now();
        const MAX_AGE_MS = 2 * 60 * 60 * 1000; // 2 hours

        for (const doc of combinedDocs) {
            const callData = doc.data();
            
            // Ghost call auto-cleanup: mark stale calls as completed
            const startedAt = callData.startedAt?.toDate ? callData.startedAt.toDate().getTime() : new Date(callData.startedAt || 0).getTime();
            if (now - startedAt > MAX_AGE_MS) {
                ghostCleanupBatch.update(doc.ref, { status: 'completed', endedAt: new Date(), duration: Math.round((now - startedAt) / 1000) });
                ghostCount++;
                continue;
            }

            // Admin API: Returns ALL active calls (user filtering removed as requested)

            const call = { id: doc.id, ...callData };
            if (call.agentId && !call.agentName) {
                try {
                    const agentDoc = await db.collection('agents').doc(call.agentId).get();
                    if (agentDoc.exists) call.agentName = agentDoc.data().name;
                } catch (e) { /* ignore */ }
            }
            calls.push({
                id: call.id,
                phoneNumber: call.phoneNumber || 'Unknown',
                crmId: call.crmId || null,
                agentId: call.agentId || '',
                agentName: call.agentName || 'Unknown Agent',
                status: call.status || 'active',
                sentiment: call.sentiment || 'neutral',
                transcript: call.transcript || '',
                transcriptMessages: call.transcriptMessages || [],
                startedAt: call.startedAt,
                endedAt: call.endedAt || null,
            });
        }

        // Auto-cleanup ghost calls in background
        if (ghostCount > 0) {
            ghostCleanupBatch.commit().catch(e => console.error('[EXT-API] Ghost cleanup error:', e));
            console.log(`[EXT-API] Auto-cleaned ${ghostCount} ghost calls`);
        }

        calls.sort((a, b) => {
            const da = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
            const db2 = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
            return db2 - da;
        });
        console.log(`[EXT-API] /v1/supervisor/calls — ${calls.length} active, ${ghostCount} ghost cleaned`);
        res.json({ success: true, data: calls, message: 'Active call fetched successfully' });
    } catch (e) {
        console.error('[EXT-API ERROR] GET /v1/supervisor/calls failed:', e);
        res.status(500).json({ success: false, data: [], message: 'Failed to fetch active calls' });
    }
});

// POST /v1/supervisor/calls/:id/whisper
router.post('/supervisor/calls/:id/whisper', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: 'message required' });

        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });

        // Ownership check: userId match OR agentId belongs to this user OR user is superadmin
        const userId = req.apiUser.userId;
        const isSuperAdmin = userId === 'superadmin-hardcoded-id';
        let owned = isSuperAdmin || call.userId === userId;
        if (!owned && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            owned = agentDoc.exists && agentDoc.data().userId === userId;
        }
        if (!owned) return res.status(404).json({ error: 'Call not found' });

        import('../index.js').then(({ broadcastToCall }) => {
            broadcastToCall(req.params.id, { type: 'whisper', message, ts: Date.now() });
        });

        const newTranscript = (call.transcript || '') + `\n[SYSTEM WHISPER]: ${message}`;
        await db.collection('calls').doc(req.params.id).update({ transcript: newTranscript });
        res.json({ success: true, message });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to whisper to AI' });
    }
});

// POST /v1/supervisor/calls/:id/barge
router.post('/supervisor/calls/:id/barge', async (req, res) => {
    try {
        const doc = await db.collection('calls').doc(req.params.id).get();
        const call = docToObj(doc);
        if (!call) return res.status(404).json({ error: 'Call not found' });
        if (call.status !== 'active') return res.status(400).json({ error: 'Cannot barge into a call that is not active' });

        // Ownership check: userId match OR agentId belongs to this user OR user is superadmin
        const userId = req.apiUser.userId;
        const isSuperAdmin = userId === 'superadmin-hardcoded-id';
        let owned = isSuperAdmin || call.userId === userId;
        if (!owned && call.agentId) {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            owned = agentDoc.exists && agentDoc.data().userId === userId;
        }
        if (!owned) return res.status(404).json({ error: 'Call not found' });

        import('../index.js').then(({ broadcastToCall }) => {
            broadcastToCall(req.params.id, { type: 'barge', ts: Date.now() });
        });

        await db.collection('calls').doc(req.params.id).update({ status: 'transferred' });
        await db.collection('systemEvents').add({ type: 'call.barged', message: `API user barged into call ${req.params.id}`, severity: 'warning', meta: '{}', createdAt: new Date() });
        res.json({ success: true });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to barge into call' });
    }
});

// ═══════════════════════════════════════════════
// DASHBOARD API
// ═══════════════════════════════════════════════

// GET /v1/dashboard/kpis
router.get('/dashboard/kpis', async (req, res) => {
    try {
        const agentsSnap = await db.collection('agents').where('userId', '==', req.apiUser.userId).get();
        const agentIds = agentsSnap.docs.map(d => d.id);

        let activeCalls = 0, completedToday = 0, allCalls = [], queueDepth = 0;
        const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);

        if (agentIds.length > 0) {
            const chunks = [];
            for (let i = 0; i < agentIds.length; i += 30) chunks.push(agentIds.slice(i, i + 30));

            for (const chunk of chunks) {
                const callsSnap = await db.collection('calls').where('agentId', 'in', chunk).get();
                callsSnap.forEach(d => {
                    const c = d.data();
                    if (c.userId && c.userId !== req.apiUser.userId) return; // safety check
                    allCalls.push(c);
                    if (c.status === 'active') activeCalls++;
                    if (c.status === 'completed' && c.startedAt && new Date(c.startedAt.toDate ? c.startedAt.toDate() : c.startedAt) >= todayStart) completedToday++;
                });
            }
        }

        const queueSnap = await db.collection('calls').where('status', '==', 'active').where('userId', '==', req.apiUser.userId).get();
        queueSnap.forEach(d => { if (!d.data().agentId) queueDepth++; });

        const avgMOS = allCalls.filter(c => c.mosScore).reduce((a, b, _, arr) => a + b.mosScore / arr.length, 0) || 4.2;
        const angryCount = allCalls.filter(c => c.sentiment === 'angry').length;
        const slaRate = allCalls.length > 0 ? Math.round((1 - angryCount / allCalls.length) * 100) : 100;

        res.json({
            activeCalls, completedToday,
            avgMOS: Math.round(avgMOS * 100) / 100,
            slaPercent: slaRate,
            apiFallbackRate: 0.5,
            aiAgentsAvailable: agentIds.length,
            humanAgentsAvailable: 2,
            queueDepth,
        });
    } catch (e) {
        console.error('[EXTERNAL API ERROR]', e);
        res.status(500).json({ error: 'Failed to metrics' });
    }
});

// ═══════════════════════════════════════════════
// DEBUG API
// ═══════════════════════════════════════════════

// GET /v1/debug/my-identity — Shows which userId is linked to the API key
router.get('/debug/my-identity', async (req, res) => {
    try {
        const apiUserId = req.apiUser.userId;
        const agentsSnap = await db.collection('agents').where('userId', '==', apiUserId).get();
        const agentIds = agentsSnap.docs.map(d => d.id);
        const callsSnap = await db.collection('calls').where('userId', '==', apiUserId).get();

        res.json({
            userId: apiUserId,
            env: req.apiUser.env,
            keyId: req.apiUser.keyId,
            ownedAgents: agentIds.length,
            agentIds,
            callsWithUserId: callsSnap.size,
            message: 'If ownedAgents is 0, your API key may be linked to the wrong user account.'
        });
    } catch (e) {
        console.error('[EXT-API ERROR] GET /v1/debug/my-identity failed:', e);
        res.status(500).json({ error: 'Failed to retrieve identity info' });
    }
});

// ═══════════════════════════════════════════════
// DISPOSITIONS CRUD API
// ═══════════════════════════════════════════════

// GET /v1/dispositions — List all dispositions for this account
router.get('/dispositions', async (req, res) => {
    try {
        const userId = req.apiUser.userId;
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 50;

        const snap = await db.collection('dispositions').get();
        let dispositions = snap.docs.map(d => ({ 
            id: d.id, 
            ...d.data(),
            linkedAgents: d.data().linkedAgents || [],
            requiredFields: d.data().requiredFields || []
        }));
        
        // Filter: show user's own + global dispositions
        dispositions = dispositions.filter(d => !d.userId || d.userId === userId);
        
        // Sort by name
        dispositions.sort((a, b) => (a.name || '').localeCompare(b.name || ''));

        // Filter: linkedAgent
        if (req.query.linkedAgent) {
            dispositions = dispositions.filter(d => 
                d.linkedAgents && d.linkedAgents.includes(req.query.linkedAgent)
            );
        }

        // Optional pagination bypass
        if (req.query.pagination === 'false') {
            return res.json(dispositions);
        }

        const total = dispositions.length;
        const startIndex = (page - 1) * limit;
        const endIndex = page * limit;
        const paginatedDispositions = dispositions.slice(startIndex, endIndex);

        res.json({ 
            dispositions: paginatedDispositions,
            pagination: { page, limit, total, totalPages: Math.ceil(total / limit) }
        });
    } catch (e) {
        console.error('[EXT-API ERROR] GET /v1/dispositions:', e);
        res.status(500).json({ error: 'Failed to list dispositions' });
    }
});

// POST /v1/dispositions — Create a new disposition
router.post('/dispositions', async (req, res) => {
    try {
        const { name, category, requiresNote, tagline, requiredFields, linkedAgents, linkedCampaigns, callDirection } = req.body;
        if (!name) return res.status(400).json({ error: "Disposition 'name' is required." });

        if (!linkedAgents || !Array.isArray(linkedAgents) || linkedAgents.length === 0) {
            return res.status(400).json({ error: "You must select at least one Agent to link this disposition to." });
        }

        const dispData = {
            name,
            category: category || 'General',
            requiresNote: requiresNote || false,
            tagline: tagline || '',
            requiredFields: Array.isArray(requiredFields) ? requiredFields : [],
            linkedAgents,
            linkedCampaigns: Array.isArray(linkedCampaigns) ? linkedCampaigns : [],
            callDirection: callDirection || 'both',
            active: true,
            userId: req.apiUser.userId,
            createdAt: new Date(),
            updatedAt: new Date()
        };

        const ref = await db.collection('dispositions').add(dispData);
        res.status(201).json({ message: 'Disposition created successfully', id: ref.id, ...dispData });
    } catch (e) {
        console.error('[EXT-API ERROR] POST /v1/dispositions:', e);
        res.status(500).json({ error: 'Failed to create disposition' });
    }
});

// GET /v1/dispositions/:id — Get a single disposition by ID
router.get('/dispositions/:id', async (req, res) => {
    try {
        const doc = await db.collection('dispositions').doc(req.params.id).get();
        if (!doc.exists) return res.status(404).json({ error: 'Disposition not found' });

        const disposition = { 
            id: doc.id, 
            ...doc.data(),
            linkedAgents: doc.data().linkedAgents || [],
            requiredFields: doc.data().requiredFields || []
        };
        // Verify ownership: allow if global or user's own
        if (disposition.userId && disposition.userId !== req.apiUser.userId) {
            return res.status(403).json({ error: 'Access denied' });
        }
        res.json(disposition);
    } catch (e) {
        console.error('[EXT-API ERROR] GET /v1/dispositions/:id:', e);
        res.status(500).json({ error: 'Failed to get disposition' });
    }
});

// PUT /v1/dispositions/:id — Update a disposition
router.put('/dispositions/:id', async (req, res) => {
    try {
        const doc = await db.collection('dispositions').doc(req.params.id).get();
        if (!doc.exists) return res.status(404).json({ error: 'Disposition not found' });

        const existing = doc.data();
        // Only allow updating own dispositions
        if (existing.userId && existing.userId !== req.apiUser.userId) {
            return res.status(403).json({ error: 'Access denied' });
        }

        const { name, category, requiresNote, active } = req.body;
        const updates = { updatedAt: new Date() };
        if (name !== undefined) updates.name = name;
        if (category !== undefined) updates.category = category;
        if (requiresNote !== undefined) updates.requiresNote = requiresNote;
        if (active !== undefined) updates.active = active;

        await db.collection('dispositions').doc(req.params.id).update(updates);
        res.json({ id: req.params.id, ...existing, ...updates });
    } catch (e) {
        console.error('[EXT-API ERROR] PUT /v1/dispositions/:id:', e);
        res.status(500).json({ error: 'Failed to update disposition' });
    }
});

// DELETE /v1/dispositions/:id — Delete a disposition
router.delete('/dispositions/:id', async (req, res) => {
    try {
        const doc = await db.collection('dispositions').doc(req.params.id).get();
        if (!doc.exists) return res.status(404).json({ error: 'Disposition not found' });

        const existing = doc.data();
        // Only allow deleting own dispositions
        if (existing.userId && existing.userId !== req.apiUser.userId) {
            return res.status(403).json({ error: 'Access denied' });
        }

        await db.collection('dispositions').doc(req.params.id).delete();
        res.json({ message: 'Disposition deleted successfully', id: req.params.id });
    } catch (e) {
        console.error('[EXT-API ERROR] DELETE /v1/dispositions/:id:', e);
        res.status(500).json({ error: 'Failed to delete disposition' });
    }
});

export default router;
