import { Router } from 'express';
import { db, docToObj } from '../firebase.js';
import WebSocket from 'ws';
import { PassThrough } from 'stream';

const router = Router();

const predictiveCache = new Map();

// Helper to clean up old speculative cache entries
setInterval(() => {
    const now = Date.now();
    for (const [key, value] of predictiveCache.entries()) {
        if (now - value.timestamp > 15000) { // Keep cache for exactly 15s
            predictiveCache.delete(key);
        }
    }
}, 10000);

// POST /api/simulation/agent-chat-predictive
// Strictly queries CX LLM in the background without touching CX Voice Engine TTS to save costs.
router.post('/agent-chat-predictive', async (req, res) => {
    try {
        const { agentId, history = [], userText, agentOverride = {}, predictionId } = req.body;
        if (!predictionId) return res.status(400).json({ error: 'Missing predictionId' });

        const agentDoc = await db.collection('agents').doc(agentId).get();
        if (!agentDoc.exists) return res.status(404).json({ error: 'Agent not found' });
        
        const dbAgent = docToObj(agentDoc);
        const agent = { ...dbAgent, ...agentOverride };

        let baseSystemPrompt = agent.systemPrompt || 'You are a helpful customer support agent.';
        const langCode = agent.language || 'en-US';
        const isHindi = langCode.startsWith('hi');
        const langConstraint = isHindi ? 'Hindi (conversational/Devanagari)' : langCode;
        
        const systemPrompt = `${baseSystemPrompt}

IMPORTANT CONVERSATIONAL PSYCHOLOGY INSTRUCTIONS (STRICT COMPLIANCE FOR VOICE TTS):
1. **Language:** Respond entirely in ${langConstraint}.
2. **Formatting:** ABSOLUTELY NO markdown, emojis, asterisks (like *laughs*), or action descriptors.
3. **Hyper-Realism on Short Answers:** If the user gives a short agreement or simple phrase (e.g., "yes", "okay", "yeah", "hello", "got it"), you MUST start your reply with a natural human backchannel (e.g., "Got it,", "Great,", "Right,", "Okay perfect-") and smoothly continue.
4. **Length Constraint:** DO NOT speak in long paragraphs or essays. Speak strictly in short, human-like 1-to-3 sentence bursts. People on the phone don't monologue!
5. **Fillers:** Use natural conversational filler words ("Well,", "Actually,", "So,") occasionally to sound completely human.
6. **Anti-Hallucination Strictness:** NEVER invent facts, prices, policies, or agree to things outside your instructions. If you don't know the answer based strictly on your context, politely say "I actually don't have that information right now." Do not guess.`;

        let contents = [];
        if (history.length > 0 && (history[0].role === 'model' || history[0].role === 'assistant')) {
            contents.push({ role: 'user', parts: [{ text: "Call connected." }] });
        }
        history.forEach(msg => {
            const mappedRole = msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user';
            if (contents.length > 0 && contents[contents.length - 1].role === mappedRole) {
                contents[contents.length - 1].parts[0].text += `\n${msg.text}`;
            } else {
                contents.push({ role: mappedRole, parts: [{ text: msg.text }] });
            }
        });
        if (userText) {
            if (contents.length > 0 && contents[contents.length - 1].role === 'user') {
                contents[contents.length - 1].parts[0].text += `\n${userText}`;
            } else {
                contents.push({ role: 'user', parts: [{ text: userText }] });
            }
        }

        const pass = new PassThrough({ objectMode: true });
        predictiveCache.set(predictionId, { timestamp: Date.now(), pass });

        // Instantly return success to the browser so it keeps listening
        res.json({ success: true, predictionId });

        const cxLlmKey = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env[Buffer.from('R0VNSU5JX0FQSV9LRVk=', 'base64').toString()];
        if (cxLlmKey && cxLlmKey !== 'MISSING_KEY') {
            const { getCXModelClient } = await import('../_rctx.js');
            const ai = await getCXModelClient(cxLlmKey);
            
            try {
                const responseStream = await ai.models.generateContentStream({
                    model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                    contents,
                    config: {
                        systemInstruction: systemPrompt,
                        temperature: agent.temperature || 0.3, // Low temperature strictly prevents hallucination
                        maxOutputTokens: agent.maxOutputTokens || 150
                    }
                });
                
                for await (const chunk of responseStream) {
                    if (chunk.text && predictiveCache.has(predictionId)) {
                        pass.write(chunk.text);
                    }
                }
                pass.end();
            } catch (e) {
                console.error("LLM predictive error:", e);
                pass.end(); 
            }
        } else {
            pass.write("Training module will be completed soon.");
            pass.end();
        }
    } catch (e) {
        console.error("Predictive stream error:", e);
    }
});

// POST /api/simulation/agent-chat-stream
router.post('/agent-chat-stream', async (req, res) => {
    try {
        const { agentId, history = [], userText, agentOverride = {}, predictionId } = req.body;

        const agentDoc = await db.collection('agents').doc(agentId).get();
        if (!agentDoc.exists) {
            return res.status(404).json({ error: 'Agent not found' });
        }
        
        const dbAgent = docToObj(agentDoc);
        const agent = { ...dbAgent, ...agentOverride };

        let baseSystemPrompt = agent.systemPrompt || 'You are a helpful customer support agent.';
        const langCode = agent.language || 'en-US';
        const isHindi = langCode.startsWith('hi');
        const langConstraint = isHindi ? 'Hindi (conversational/Devanagari)' : langCode;
        
        const systemPrompt = `${baseSystemPrompt}

IMPORTANT CONVERSATIONAL PSYCHOLOGY INSTRUCTIONS (STRICT COMPLIANCE FOR VOICE TTS):
1. **Language:** Respond entirely in ${langConstraint}.
2. **Formatting:** ABSOLUTELY NO markdown, emojis, asterisks (like *laughs*), or action descriptors.
3. **Hyper-Realism on Short Answers:** If the user gives a short agreement or simple phrase (e.g., "yes", "okay", "yeah", "hello", "got it"), you MUST start your reply with a natural human backchannel (e.g., "Got it,", "Great,", "Right,", "Okay perfect-") and smoothly continue.
4. **Length Constraint:** DO NOT speak in long paragraphs or essays. Speak strictly in short, human-like 1-to-3 sentence bursts. People on the phone don't monologue!
5. **Fillers:** Use natural conversational filler words ("Well,", "Actually,", "So,") occasionally to sound completely human.
6. **Anti-Hallucination Strictness:** NEVER invent facts, prices, policies, or agree to things outside your instructions. If you don't know the answer based strictly on your context, politely say "I actually don't have that information right now." Do not guess.`;


        const voiceId = agent.voice || 'MF4J4IDTRo0AxOO4dpFR';
        const prosodyRate = agent.prosodyRate ?? 1.0;
        const prosodyPitch = agent.prosodyPitch ?? 1.0;

        let contents = [];
        if (history.length === 0 && !userText) {
            const greetingPrompt = isHindi
                ? `You are initiating an outbound phone call. Based on your system instructions, generate the optimal natural Hindi opening line to greet the customer the moment they pick up. Be convincing, direct, and conversational. Output strictly what you say.`
                : `You are initiating an outbound phone call. Based on your system instructions, generate a natural opening line to greet the customer. Do not include pleasantries or actions, just what you say.`;
            contents.push({ role: 'user', parts: [{ text: greetingPrompt }] });
        } else {
            if (history.length > 0 && (history[0].role === 'model' || history[0].role === 'assistant')) {
                contents.push({ role: 'user', parts: [{ text: "Call connected." }] });
            }
            history.forEach(msg => {
                const mappedRole = msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user';
                if (contents.length > 0 && contents[contents.length - 1].role === mappedRole) {
                    contents[contents.length - 1].parts[0].text += `\n${msg.text}`;
                } else {
                    contents.push({ role: mappedRole, parts: [{ text: msg.text }] });
                }
            });
            if (userText) {
                if (contents.length > 0 && contents[contents.length - 1].role === 'user') {
                    contents[contents.length - 1].parts[0].text += `\n${userText}`;
                } else {
                    contents.push({ role: 'user', parts: [{ text: userText }] });
                }
            }
        }

        let stability = 0.5, similarity = 0.5;
        if (prosodyRate > 1.2 || prosodyPitch > 1.2) stability = 0.3;
        if (prosodyRate < 0.8 || prosodyPitch < 0.8) stability = 0.8;

        const legacyVoiceMap = { 'alloy': 'MF4J4IDTRo0AxOO4dpFR', 'echo': '1qEiC6qsybMkmnNdVMbK', 'fable': 'qDuRKMlYmrm8trt5QyBn', 'onyx': 'LQ2auZHpAQ9h4azztqMT', 'nova': 's6cZdgI3j07hf4frz4Q8', 'shimmer': 'MF4J4IDTRo0AxOO4dpFR' };
        const resolvedVoiceId = legacyVoiceMap[voiceId] || voiceId;

        // Set Headers for streaming HTTP response back to browser
        res.setHeader('Content-Type', 'application/x-ndjson');
        res.setHeader('Transfer-Encoding', 'chunked');

        const voiceApiKey = process.env.CALLEX_VOICE_API_KEY || '030a62b112af48f06748c478cd7f607c386f41b30d1be8ffc680484f808a6d9c';
        const _voiceWsBase = Buffer.from('d3NzOi8vYXBpLmVsZXZlbmxhYnMuaW8vdjEvdGV4dC10by1zcGVlY2gv', 'base64').toString('utf-8');
        const wsUrl = `${_voiceWsBase}${resolvedVoiceId}/stream-input?model_id=eleven_multilingual_v2`;
        
        const voiceWs = new WebSocket(wsUrl);

        let isWsOpen = false;
        let pendingTextChunks = [];
        let fullAiText = "";

        voiceWs.on('open', () => {
            isWsOpen = true;
            
            // Send initial configuration with the first space to initialize connection
            const configMsg = {
                text: " ",
                voice_settings: { stability, similarity_boost: similarity },
                xi_api_key: voiceApiKey,
            };
            voiceWs.send(JSON.stringify(configMsg));

            // Flush any pending text chunks that LLM produced before WS opened
            let isFirst = true;
            for (const chunk of pendingTextChunks) {
                voiceWs.send(JSON.stringify({ text: chunk, flush: isFirst }));
                isFirst = false;
            }
            pendingTextChunks = [];
        });

        // Pipe CX Voice Engine audio directly to the Express response
        voiceWs.on('message', (data) => {
            try {
                const msg = JSON.parse(data);
                if (msg.audio) {
                    res.write(JSON.stringify({ type: 'audio', data: msg.audio }) + '\n');
                }
                if (msg.isFinal) {
                    voiceWs.close();
                }
            } catch (err) {
                console.error("CX Voice WS message parse error:", err);
            }
        });

        voiceWs.on('error', (err) => {
            console.error("CX Voice WS Error:", err);
            if (!res.headersSent) res.status(500).end();
            else res.end();
        });

        voiceWs.on('close', () => {
            res.end();
        });

        // ── Connect to CX LLM via Stream ──
        const cxLlmKey = process.env.GENARTML_SERVER_KEY || process.env.CX_LLM_KEY || process.env[Buffer.from('R0VNSU5JX0FQSV9LRVk=', 'base64').toString()] || 'MISSING_KEY';
        
        let responseIterator;
        if (predictionId && predictiveCache.has(predictionId)) {
            console.log("⚡ SPECULATIVE CACHE HIT:", predictionId);
            responseIterator = predictiveCache.get(predictionId).pass;
            predictiveCache.delete(predictionId);
        } else if (cxLlmKey && cxLlmKey !== 'MISSING_KEY') {
            const { getCXModelClient } = await import('../_rctx.js');
            const ai = await getCXModelClient(cxLlmKey);
            
            try {
                const responseStream = await ai.models.generateContentStream({
                    model: Buffer.from('Z2VtaW5pLTIuNS1mbGFzaA==', 'base64').toString(),
                    contents,
                    config: {
                        systemInstruction: systemPrompt,
                        temperature: agent.temperature || 0.3, // Low temperature strictly prevents hallucination
                        maxOutputTokens: agent.maxOutputTokens || 150
                    }
                });

                // Wrap responseStream into an iterator of strings
                responseIterator = (async function*() {
                    for await (const chunk of responseStream) {
                        if (chunk.text) yield chunk.text;
                    }
                })();
            } catch (e) {
                console.error("LLM Streaming Error:", e);
                if (isWsOpen) voiceWs.send(JSON.stringify({ text: "" }));
                return res.end();
            }
        } else {
            // Fallback
            responseIterator = (async function*() {
                yield "Training module will be completed soon.";
            })();
        }

        try {
            let textBuffer = '';
            let isFirstFlush = true;

            for await (const chunkText of responseIterator) {
                if (chunkText) {
                    fullAiText += chunkText;
                    textBuffer += chunkText;
                    
                    // Stream the original text immediately to the frontend so it retains context
                    res.write(JSON.stringify({ type: 'text', data: chunkText }) + '\n');
                        
                        // Check for natural sentence/clause boundaries
                        // We use a regex match on the buffer to wait for punctuation.
                        let match;
                        while ((match = textBuffer.match(/^(.*?[.,:;!?\n-])\s*(.*)$/s))) {
                            const clause = match[1] + " "; // Add a space for natural pause
                            textBuffer = match[2]; // Keep remainder
                            
                            if (isWsOpen) {
                                voiceWs.send(JSON.stringify({ 
                                    text: clause, 
                                    flush: isFirstFlush // Only force flush the first clause to drop TTFB instantly!
                                }));
                                isFirstFlush = false;
                            } else {
                                pendingTextChunks.push(clause);
                            }
                        }

                        // Fallback: If buffer gets too large without punctuation (e.g., >80 chars), flush a word boundary
                        if (textBuffer.length > 80 && textBuffer.includes(" ")) {
                            const lastSpace = textBuffer.lastIndexOf(" ");
                            const clause = textBuffer.substring(0, lastSpace) + " ";
                            textBuffer = textBuffer.substring(lastSpace + 1);
                            
                            if (isWsOpen) {
                                voiceWs.send(JSON.stringify({ text: clause }));
                            } else {
                                pendingTextChunks.push(clause);
                            }
                        }
                    } // close if (chunkText)
                } // close for await

                // Stream is done, send remaining buffer
                if (textBuffer.trim().length > 0) {
                    if (isWsOpen) {
                        voiceWs.send(JSON.stringify({ text: textBuffer }));
                    } else {
                        pendingTextChunks.push(textBuffer);
                    }
                }

                // Signal end of text stream to CX Voice Engine
                if (isWsOpen) {
                    voiceWs.send(JSON.stringify({ text: "" }));
                } else {
                    pendingTextChunks.push(""); // The empty string signals the end to CX Voice Engine
                }

            } catch (e) {
                console.error("Stream parsing error:", e);
                if (isWsOpen) voiceWs.send(JSON.stringify({ text: "" }));
            }

    } catch (e) {
        console.error("Simulation Stream Error:", e);
        if (!res.headersSent) res.status(500).json({ error: 'Failed to stream response' });
        else res.end();
    }
});

export default router;
