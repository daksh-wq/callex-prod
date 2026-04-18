import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import multer from 'multer';

const router = Router();
const upload = multer({ storage: multer.memoryStorage() });

// GET /api/agents
router.get('/', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 50;

        const snap = await db.collection('agents').where('userId', '==', req.userId).get();
        let agents = queryToArray(snap);
        agents.sort((a, b) => {
            const ta = a.createdAt?.toDate ? a.createdAt.toDate().getTime() : new Date(a.createdAt || 0).getTime();
            const tb = b.createdAt?.toDate ? b.createdAt.toDate().getTime() : new Date(b.createdAt || 0).getTime();
            return tb - ta;
        });

        if (req.query.pagination === 'false') {
            return res.json(agents);
        }

        const total = agents.length;
        const startIndex = (page - 1) * limit;
        const endIndex = page * limit;
        const paginatedAgents = agents.slice(startIndex, endIndex);

        res.json({
            agents: paginatedAgents,
            pagination: { page, limit, total, totalPages: Math.ceil(total / limit) }
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/agents/:id
router.get('/:id', async (req, res) => {
    const doc = await db.collection('agents').doc(req.params.id).get();
    const agent = docToObj(doc);
    if (!agent || agent.userId !== req.userId) return res.status(404).json({ error: 'Agent not found' });
    // Get prompt versions
    const pvSnap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
    const versions = queryToArray(pvSnap);
    versions.sort((a, b) => (b.version || 0) - (a.version || 0));
    agent.PromptVersion = versions;
    res.json(agent);
});

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

// POST /api/agents
router.post('/', upload.single('file'), async (req, res) => {
  try {
    const { name, description, systemPrompt, openingLine, voice, language, sttEngine, llmModel,
        fillerPhrases, prosodyRate, prosodyPitch, ipaLexicon, tools, topK, similarityThresh,
        fallbackMessage, profanityFilter, topicRestriction, backgroundAmbience, speakingStyle,
        bargeInMode, patienceMs, maxDuration, temperature, maxTokens, strictToolCalling, backgroundNoiseVolume,
        ringTimeout, voicemailLogic, webhookUrl, autoSummary, autoSentiment, recordCall, processDtmf,
        amdPrecision, voicemailDropAudio, sentimentRouting, competitorAlerts, supervisorWhisper,
        piiRedaction, geoCallerId, multiAgentHandoff, objectionHandling, emotionalMirroring,
        complianceScript, dynamicCodeSwitching, dncLitigatorScrub, callBlending, costCapTokens,
        postCallSms, autoFollowUp, followUpDefaultDays, followUpDefaultTime, analysisSchema, dispositions,
        voiceSpeed
    } = req.body;

    if (!name) return res.status(400).json({ error: "Agent 'name' is required." });

    const data = {
        userId: req.userId,
        name, description: description || '', systemPrompt: systemPrompt || '', openingLine: openingLine || '',
        voice: voice || 'alloy', language: language || 'en-US', sttEngine: sttEngine || 'callex-1.1',
        llmModel: llmModel || 'callex-1.3',
        fillerPhrases: typeof fillerPhrases === 'string' ? fillerPhrases : JSON.stringify(fillerPhrases || ['Let me check...', 'One moment...']),
        prosodyRate: parseNum(prosodyRate, 1.0), prosodyPitch: parseNum(prosodyPitch, 1.0),
        voiceSpeed: parseNum(voiceSpeed, 1.0),
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
        backgroundNoiseVolume: parseNum(backgroundNoiseVolume, 0.20),
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
        const { buffer, mimetype, originalname, size } = req.file;
        const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env.$(echo R0VNSU5JX0FQSV9LRVk= | base64 -d);
        const ext = '.' + originalname.split('.').pop().toLowerCase();
        
        if (CX_LLM_KEY) {
            const { getCXModelClient } = await import('../_rctx.js');
            const genAI = await getCXModelClient(CX_LLM_KEY);

            const isTextFile = mimetype === 'text/plain' || mimetype === 'text/csv' || ext === '.csv' || ext === '.txt';
            let knowledgeText = '';

            try {
                if (isTextFile) {
                    const rawText = buffer.toString('utf-8');
                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{ role: 'user', parts: [{ text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nContent:\n${rawText}` }] }]
                    });
                    knowledgeText = response.text;
                } else {
                    const base64Data = buffer.toString('base64');
                    let docMimeType = mimetype;
                    if (ext === '.xlsx') docMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
                    if (ext === '.xls') docMimeType = 'application/vnd.ms-excel';
                    if (ext === '.pdf') docMimeType = 'application/pdf';

                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{ role: 'user', parts: [
                            { inlineData: { mimeType: docMimeType, data: base64Data } },
                            { text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nRules: Extract everything. Convert pricing to spoken format. Keep the original language.` }
                        ] }]
                    });
                    knowledgeText = response.text;
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
            } catch (err) {
                console.error('[KNOWLEDGE EXTRACTION ERROR]', err);
            }
        }
    }

    const ref = await db.collection('agents').add(data);
    const agent = { id: ref.id, ...data };

    await db.collection('promptVersions').add({
        agentId: ref.id, version: 1, prompt: systemPrompt || '', isActive: true, label: 'v1 - Initial', createdAt: new Date()
    });

    // --- Create Shadow Training Sandbox Automatically ---
    if (!data.isTrainingSandbox) {
        const sandboxData = {
            ...data,
            name: `${data.name} - Training Sandbox`,
            isTrainingSandbox: true,
            parentAgentId: ref.id,
            status: 'draft',
            createdAt: new Date(),
            updatedAt: new Date()
        };
        try {
            const sandboxRef = await db.collection('agents').add(sandboxData);
            await db.collection('promptVersions').add({
                agentId: sandboxRef.id, version: 1, prompt: systemPrompt || '', isActive: true, label: 'v1 - Sandbox Initial', createdAt: new Date()
            });
            console.log(`[AGENT CREATION] Auto-generated Shadow Sandbox: ${sandboxRef.id}`);
        } catch (e) {
            console.error('[AGENT CREATION] Failed to generate Shadow Sandbox:', e);
        }
    }

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
                        userId: req.userId,
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

    res.json(agent);
  } catch (err) {
    console.error('[AGENT CREATION ERROR]', err);
    res.status(500).json({ error: err.message || 'Failed to create agent' });
  }
});

// PATCH /api/agents/:id
router.patch('/:id', upload.single('file'), async (req, res) => {
    const doc = await db.collection('agents').doc(req.params.id).get();
    const existing = docToObj(doc);
    if (!existing || existing.userId !== req.userId) return res.status(404).json({ error: 'Agent not found' });

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
    const numFields = ['prosodyRate', 'prosodyPitch', 'topK', 'similarityThresh', 'patienceMs', 'maxDuration', 'temperature', 'maxTokens', 'ringTimeout', 'costCapTokens', 'followUpDefaultDays', 'voiceSpeed', 'backgroundNoiseVolume'];
    for (const f of numFields) {
        if (updates[f] !== undefined) updates[f] = parseNum(updates[f], updates[f]);
    }

    // Handle Knowledge Base Upload if file is present
    if (req.file) {
        const { buffer, mimetype, originalname } = req.file;
        const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env.$(echo R0VNSU5JX0FQSV9LRVk= | base64 -d);
        const ext = '.' + originalname.split('.').pop().toLowerCase();
        
        if (CX_LLM_KEY) {
            const { getCXModelClient } = await import('../_rctx.js');
            const genAI = await getCXModelClient(CX_LLM_KEY);
            const isTextFile = mimetype === 'text/plain' || mimetype === 'text/csv' || ext === '.csv' || ext === '.txt';
            let knowledgeText = '';

            try {
                if (isTextFile) {
                    const rawText = buffer.toString('utf-8');
                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{ role: 'user', parts: [{ text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nContent:\n${rawText}` }] }]
                    });
                    knowledgeText = response.text;
                } else {
                    const base64Data = buffer.toString('base64');
                    let docMimeType = mimetype;
                    if (ext === '.xlsx') docMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
                    if (ext === '.xls') docMimeType = 'application/vnd.ms-excel';
                    if (ext === '.pdf') docMimeType = 'application/pdf';

                    const response = await genAI.models.generateContent({
                        model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                        contents: [{ role: 'user', parts: [
                            { inlineData: { mimeType: docMimeType, data: base64Data } },
                            { text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nRules: Extract everything. Convert pricing to spoken format. Keep the original language.` }
                        ] }]
                    });
                    knowledgeText = response.text;
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
            } catch (err) {
                console.error('[KNOWLEDGE EXTRACTION ERROR]', err);
            }
        }
    }

    // Handle system prompt clear or update dynamically outside of the prompt tab
    if (updates.systemPrompt !== undefined) {
        try {
            // Fetch ALL prompt versions for this agent (no orderBy = no composite index needed)
            const allPvSnap = await db.collection('promptVersions')
                .where('agentId', '==', req.params.id).get();
            
            let nextVersion = 1;
            if (!allPvSnap.empty) {
                // Find highest version number in memory (avoids needing Firebase composite index)
                let maxVersion = 0;
                for (const pvDoc of allPvSnap.docs) {
                    const v = pvDoc.data().version || 0;
                    if (v > maxVersion) maxVersion = v;
                    // Deactivate all existing versions
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
            console.log(`[AGENT EDIT] ✅ Prompt version v${nextVersion} saved for agent ${req.params.id}`);
        } catch(err) {
            console.error('[AGENT EDIT] Failed to save prompt version:', err);
        }
    }

    delete updates.id;

    await db.collection('agents').doc(req.params.id).update(updates);
    const updated = await db.collection('agents').doc(req.params.id).get();
    res.json(docToObj(updated));
});

// DELETE /api/agents/:id
router.delete('/:id', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.userId) return res.status(404).json({ error: 'Agent not found' });

        // Delete related records to prevent orphaned data
        const pvSnap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
        const fuSnap = await db.collection('followUps').where('agentId', '==', req.params.id).get();
        
        const batch = db.batch();
        pvSnap.forEach(d => batch.delete(d.ref));
        fuSnap.forEach(d => batch.delete(d.ref));
        batch.delete(db.collection('agents').doc(req.params.id));
        
        await batch.commit();

        res.json({ success: true });
    } catch (e) {
        console.error('[AGENTS] Delete error:', e);
        res.status(500).json({ error: 'Failed to delete agent' });
    }
});

// POST /api/agents/:id/knowledge — Upload document to train agent
router.post('/:id/knowledge', upload.single('file'), async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.userId) return res.status(404).json({ error: 'Agent not found' });

        if (!req.file) return res.status(400).json({ error: 'No file uploaded. Send a file with field name "file".' });

        const { buffer, mimetype, originalname, size } = req.file;
        const allowedExtensions = ['.pdf', '.xlsx', '.xls', '.csv', '.txt'];
        const ext = '.' + originalname.split('.').pop().toLowerCase();

        if (!allowedExtensions.includes(ext)) {
            return res.status(400).json({ error: 'Unsupported file type. Allowed: PDF, Excel, CSV, TXT' });
        }

        console.log(`[KNOWLEDGE] Processing ${originalname} (${(size / 1024).toFixed(1)}KB) for agent ${req.params.id}`);

        const CX_LLM_KEY = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env.$(echo R0VNSU5JX0FQSV9LRVk= | base64 -d);
        if (!CX_LLM_KEY) return res.status(500).json({ error: 'AI API key not configured' });

        const { getCXModelClient } = await import('../_rctx.js');
        const genAI = await getCXModelClient(CX_LLM_KEY);

        // Use AI to parse ANY file type intelligently
        const isTextFile = mimetype === 'text/plain' || mimetype === 'text/csv' || ext === '.csv' || ext === '.txt';
        let knowledgeText = '';

        if (isTextFile) {
            const rawText = buffer.toString('utf-8');
            const response = await genAI.models.generateContent({
                model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                contents: [{ role: 'user', parts: [{ text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nContent:\n${rawText}` }] }]
            });
            knowledgeText = response.text;
        } else {
            const base64Data = buffer.toString('base64');
            let docMimeType = mimetype;
            if (ext === '.xlsx') docMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
            if (ext === '.xls') docMimeType = 'application/vnd.ms-excel';
            if (ext === '.pdf') docMimeType = 'application/pdf';

            const response = await genAI.models.generateContent({
                model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                contents: [{ role: 'user', parts: [
                    { inlineData: { mimeType: docMimeType, data: base64Data } },
                    { text: `You are a Knowledge Extractor for an AI calling agent. Extract ALL useful information.\n\nOutput format:\nKNOWLEDGE BASE:\n[All Q&A pairs, facts, pricing, policies etc.]\n\nTOPICS COVERED:\n[Comma-separated topics]\n\nTOTAL ITEMS:\n[Number]\n\nSAMPLE QUESTIONS:\n[5 customer questions]\n\nRules: Extract everything. Convert pricing to spoken format. Keep the original language.` }
                ] }]
            });
            knowledgeText = response.text;
        }

        if (!knowledgeText || knowledgeText.length < 50) {
            return res.status(422).json({ error: 'Could not extract meaningful knowledge from the file.' });
        }

        // Parse metadata
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

        // Merge with existing
        const existingKnowledge = existing.knowledgeBase || '';
        const mergedKnowledge = existingKnowledge
            ? `${existingKnowledge}\n\n--- New Knowledge (from ${originalname}) ---\n\n${extractedKnowledge}`
            : extractedKnowledge;

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

        await db.collection('agents').doc(req.params.id).update({
            knowledgeBase: mergedKnowledge,
            knowledgeTopics: trainingSummary.knowledgeTopics,
            trainingSummary,
            updatedAt: new Date(),
        });

        console.log(`[KNOWLEDGE] ✅ Agent ${req.params.id} trained with ${originalname}`);
        res.json({ message: 'Knowledge uploaded and processed successfully', trainingSummary, knowledgeSize: mergedKnowledge.length });
    } catch (e) {
        console.error('[KNOWLEDGE ERROR]', e);
        res.status(500).json({ error: 'Failed to process knowledge file', details: e.message });
    }
});

// DELETE /api/agents/:id/knowledge — Clear knowledge base
router.delete('/:id/knowledge', async (req, res) => {
    try {
        const doc = await db.collection('agents').doc(req.params.id).get();
        const existing = docToObj(doc);
        if (!existing || existing.userId !== req.userId) return res.status(404).json({ error: 'Agent not found' });

        await db.collection('agents').doc(req.params.id).update({
            knowledgeBase: '', knowledgeTopics: [], trainingSummary: null, updatedAt: new Date(),
        });
        res.json({ message: 'Knowledge base cleared successfully' });
    } catch (e) {
        console.error('[KNOWLEDGE ERROR]', e);
        res.status(500).json({ error: 'Failed to clear knowledge base' });
    }
});

// POST /api/agents/:id/prompt-version
router.post('/:id/prompt-version', async (req, res) => {
    const { prompt, label } = req.body;
    // Fetch all versions (no orderBy = no composite index needed)
    const allPvSnap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
    let lastVersion = 0;
    const batch = db.batch();
    allPvSnap.forEach(d => {
        const v = d.data().version || 0;
        if (v > lastVersion) lastVersion = v;
        batch.update(d.ref, { isActive: false });
    });
    const version = lastVersion + 1;

    const pvData = { agentId: req.params.id, version, prompt, label: label || `v${version}`, isActive: true, createdAt: new Date() };
    const pvRef = db.collection('promptVersions').doc();
    batch.set(pvRef, pvData);
    await batch.commit();

    res.json({ id: pvRef.id, ...pvData });
});

// GET /api/agents/:id/prompt-versions
router.get('/:id/prompt-versions', async (req, res) => {
    const snap = await db.collection('promptVersions').where('agentId', '==', req.params.id).get();
    const versions = queryToArray(snap).sort((a, b) => (b.version || 0) - (a.version || 0));
    res.json(versions);
});

// PATCH /api/agents/:id/status
router.patch('/:id/status', async (req, res) => {
    const { status } = req.body;
    await db.collection('agents').doc(req.params.id).update({ status, updatedAt: new Date() });
    const doc = await db.collection('agents').doc(req.params.id).get();
    res.json(docToObj(doc));
});

// POST /api/agents/tts-preview
router.post('/tts-preview', async (req, res) => {
    try {
        const { voiceId, prosodyRate, prosodyPitch } = req.body;
        if (!voiceId) return res.status(400).json({ error: 'Voice ID required' });

        let stability = 0.5, similarity = 0.5;
        if (prosodyRate > 1.2 || prosodyPitch > 1.2) stability = 0.3;
        if (prosodyRate < 0.8 || prosodyPitch < 0.8) stability = 0.8;

        const legacyVoiceMap = {
            'alloy': 'MF4J4IDTRo0AxOO4dpFR', 'echo': '1qEiC6qsybMkmnNdVMbK',
            'fable': 'qDuRKMlYmrm8trt5QyBn', 'onyx': 'LQ2auZHpAQ9h4azztqMT',
            'nova': 's6cZdgI3j07hf4frz4Q8', 'shimmer': 'MF4J4IDTRo0AxOO4dpFR'
        };
        const resolvedVoiceId = legacyVoiceMap[voiceId] || voiceId;
        const defaultVoiceId = 'MF4J4IDTRo0AxOO4dpFR';
        const ttsPayload = {
            text: "नमस्ते, मैं Callex हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
            model_id: Buffer.from('ZWxldmVuX211bHRpbGluZ3VhbF92Mg==', 'base64').toString(),
            voice_settings: { stability, similarity_boost: similarity }
        };

        const _ttsBase = Buffer.from('aHR0cHM6Ly9hcGkuZWxldmVubGFicy5pby92MS90ZXh0LXRvLXNwZWVjaC8=', 'base64').toString();
        let response = await fetch(`${_ttsBase}${resolvedVoiceId}/stream`, {
            method: 'POST',
            headers: {
                'Accept': 'audio/mpeg', 'Content-Type': 'application/json',
                'xi-api-key': process.env.CALLEX_VOICE_API_KEY || '030a62b112af48f06748c478cd7f607c386f41b30d1be8ffc680484f808a6d9c'
            },
            body: JSON.stringify(ttsPayload)
        });

        // Production fallback: if voice ID invalid, retry with default Callex voice
        if (!response.ok) {
            console.log(`[Callex Voice Engine] Voice ${resolvedVoiceId} failed (${response.status}), falling back to default...`);
            response = await fetch(`${_ttsBase}${defaultVoiceId}/stream`, {
                method: 'POST',
                headers: {
                    'Accept': 'audio/mpeg', 'Content-Type': 'application/json',
                    'xi-api-key': process.env.CALLEX_VOICE_API_KEY || '030a62b112af48f06748c478cd7f607c386f41b30d1be8ffc680484f808a6d9c'
                },
                body: JSON.stringify(ttsPayload)
            });
        }

        if (!response.ok) { const errText = await response.text(); throw new Error(`Voice Engine Error: ${response.status} ${errText}`); }
        res.setHeader('Content-Type', 'audio/mpeg');
        res.setHeader('Transfer-Encoding', 'chunked');
        for await (const chunk of response.body) { res.write(chunk); }
        res.end();
    } catch (e) {
        console.error("TTS Preview Error:", e);
        res.status(500).json({ error: 'Failed to generate TTS preview' });
    }
});

// POST /api/agents/clone-voice
router.post('/clone-voice', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No audio file uploaded' });
        const formData = new FormData();
        formData.append('name', req.body.name || 'Cloned Dashboard Voice');
        formData.append('description', 'Instant Voice Clone created via Agent Studio');
        const audioBlob = new Blob([req.file.buffer], { type: req.file.mimetype });
        formData.append('files', audioBlob, req.file.originalname || 'clone.mp3');

        const _voiceAddUrl = Buffer.from('aHR0cHM6Ly9hcGkuZWxldmVubGFicy5pby92MS92b2ljZXMvYWRk', 'base64').toString();
        const response = await fetch(_voiceAddUrl, {
            method: 'POST',
            headers: { 'xi-api-key': process.env.CALLEX_VOICE_API_KEY || '030a62b112af48f06748c478cd7f607c386f41b30d1be8ffc680484f808a6d9c' },
            body: formData
        });
        if (!response.ok) { const errText = await response.text(); throw new Error(`Voice Engine Error: ${response.status} ${errText}`); }
        const data = await response.json();
        res.json({ voiceId: data.voice_id });
    } catch (e) {
        console.error("Voice Cloning Error:", e);
        res.status(500).json({ error: 'Failed to clone voice' });
    }
});

// POST /api/agents/:id/push-to-prod
router.post('/:id/push-to-prod', async (req, res) => {
    try {
        const sandboxDoc = await db.collection('agents').doc(req.params.id).get();
        if (!sandboxDoc.exists) return res.status(404).json({ error: 'Sandbox Agent not found' });
        
        const sandbox = docToObj(sandboxDoc);
        if (!sandbox.isTrainingSandbox || !sandbox.parentAgentId || sandbox.userId !== req.userId) {
            return res.status(400).json({ error: 'Invalid Sandbox Agent for Production Push' });
        }

        const parentDoc = await db.collection('agents').doc(sandbox.parentAgentId).get();
        if (!parentDoc.exists) return res.status(404).json({ error: 'Parent Production Agent not found' });

        // Update the parent agent with the finalized Sandbox Prompt & KnowledgeBase
        const updates = { 
            systemPrompt: sandbox.systemPrompt, 
            knowledgeBase: sandbox.knowledgeBase,
            updatedAt: new Date() 
        };
        await db.collection('agents').doc(sandbox.parentAgentId).update(updates);

        // Versioning for the parent agent
        const pvSnap = await db.collection('promptVersions')
            .where('agentId', '==', sandbox.parentAgentId)
            .get();
        let maxVersion = 0;
        pvSnap.forEach(doc => {
            const pv = doc.data();
            if (pv.version > maxVersion) maxVersion = pv.version;
            doc.ref.update({ isActive: false });
        });
        
        await db.collection('promptVersions').add({
            agentId: sandbox.parentAgentId,
            version: maxVersion + 1,
            prompt: sandbox.systemPrompt || '',
            isActive: true,
            label: `v${maxVersion + 1} - Pushed from Sandbox`,
            createdAt: new Date()
        });

        res.json({ success: true, message: 'Successfully pushed Sandbox to Production', parentAgentId: sandbox.parentAgentId });
    } catch (e) {
        console.error("[PUSH TO PROD ERROR]", e);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

export default router;
