import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import dns from 'dns';
import path from 'path';
import { fileURLToPath } from 'url';

// Fix for Node.js 18+ native fetch ENOTFOUND errors on macOS
dns.setDefaultResultOrder('ipv4first');

// Firebase
import { db } from './firebase.js';

// Routes
import dashboardRouter from './routes/dashboard.js';
import supervisorRouter from './routes/supervisor.js';
import agentsRouter from './routes/agents.js';
import knowledgeRouter from './routes/knowledge.js';
import simulationRouter from './routes/simulation.js';
import simulationStreamRouter from './routes/simulation-stream.js';
import dialerRouter from './routes/dialer.js';
import analyticsRouter from './routes/analytics.js';
import routingRouter from './routes/routing.js';
import integrationsRouter from './routes/integrations.js';
import securityRouter from './routes/security.js';
import settingsRouter from './routes/settings.js';
import qaRouter from './routes/qa.js';
import wfmRouter from './routes/wfm.js';
import telecomRouter from './routes/telecom.js';
import billingRouter from './routes/billing.js';
import reportsRouter from './routes/reports.js';
import authRouter from './routes/auth.js';
import followupsRouter from './routes/followups.js';

import { setupSupervisorWS } from './ws/supervisor.js';
import { setupDashboardWS } from './ws/dashboard.js';
import { setupCXSTTWS } from './ws/_wst.js';

const __root_filename = fileURLToPath(import.meta.url);
const __root_dirname = path.dirname(__root_filename);
dotenv.config({ path: path.resolve(__root_dirname, '../../../.env') });

const app = express();
const httpServer = createServer(app);
export const wss = new WebSocketServer({ server: httpServer });

// Middleware
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));
app.use('/uploads', express.static('uploads'));

// Routes
app.use('/api/auth', authRouter);

// Apply JWT auth middleware to all other /api routes
import { requireAuth } from './middleware/auth.js';
app.use('/api', (req, res, next) => {
    if (req.path.startsWith('/v1')) return next();
    requireAuth(req, res, next);
});

app.use('/api/dashboard', dashboardRouter);
app.use('/api/supervisor', supervisorRouter);
app.use('/api/agents', agentsRouter);
app.use('/api/knowledge', knowledgeRouter);
app.use('/api/simulation', simulationRouter);
app.use('/api/simulation-stream', simulationStreamRouter);
app.use('/api/dialer', dialerRouter);
app.use('/api/analytics', analyticsRouter);
app.use('/api/routing', routingRouter);
app.use('/api/integrations', integrationsRouter);
app.use('/api/security', securityRouter);
app.use('/api/settings', settingsRouter);
app.use('/api/qa', qaRouter);
app.use('/api/wfm', wfmRouter);
app.use('/api/telecom', telecomRouter);
app.use('/api/billing', billingRouter);
app.use('/api/reports', reportsRouter);
app.use('/api/followups', followupsRouter);

// Admin panel (super-admin only)
import adminRouter from './routes/admin.js';
app.use('/api/admin', adminRouter);

// External developer APIs
import externalRouter from './routes/external.js';
app.use('/api/v1', externalRouter);

// WebSocket routing
export const wsClients = new Map();
export const dashboardClients = new Set();

wss.on('connection', (ws, req) => {
    const url = new URL(req.url, 'http://localhost');
    const type = url.searchParams.get('type');
    const callId = url.searchParams.get('callId');

    if (type === 'dashboard') {
        dashboardClients.add(ws);
        setupDashboardWS(ws);
        ws.on('close', () => dashboardClients.delete(ws));
    } else if (type === 'supervisor' && callId) {
        if (!wsClients.has(callId)) wsClients.set(callId, new Set());
        wsClients.get(callId).add(ws);
        setupSupervisorWS(ws, callId);
        ws.on('close', () => { wsClients.get(callId)?.delete(ws); });
    } else if (type === 'softphone') {
        ws.on('message', (msg) => {
            ws.send(JSON.stringify({ type: 'transcript', text: '[STT simulation active]', ts: Date.now() }));
        });
    } else if (type === 'stt') {
        const langCode = url.searchParams.get('lang') || 'hi-IN';
        setupCXSTTWS(ws, { languageCode: langCode });
    }
});

// Broadcast helpers
export function broadcastToDashboard(data) {
    const msg = JSON.stringify(data);
    dashboardClients.forEach(c => { if (c.readyState === 1) c.send(msg); });
}

export function broadcastToCall(callId, data) {
    const msg = JSON.stringify(data);
    wsClients.get(callId)?.forEach(c => { if (c.readyState === 1) c.send(msg); });
}

// Serve the Enterprise Dashboard frontend
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DASHBOARD_DIST = path.resolve(__dirname, '../../frontend/dist');

app.use(express.static(DASHBOARD_DIST));

// Health check
app.get('/api/health', (req, res) => res.json({ status: 'ok', ts: new Date() }));

// SPA catch-all
app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) return res.status(404).json({ error: 'Not found' });
    res.sendFile(path.join(DASHBOARD_DIST, 'index.html'));
});

const PORT = process.env.PORT || 4500;
httpServer.listen(PORT, () => {
    console.log(`[ENTERPRISE] Backend running on http://localhost:${PORT}`);
    seedInitialData();
});

async function seedInitialData() {
    try {
        // Seed integrations if empty
        const intSnap = await db.collection('integrations').limit(1).get();
        if (intSnap.empty) {
            const integrations = [
                { name: 'Salesforce', slug: 'salesforce', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Stripe', slug: 'stripe', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Zapier', slug: 'zapier', connected: false, config: '{}', createdAt: new Date() },
                { name: 'HubSpot', slug: 'hubspot', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Twilio', slug: 'twilio', connected: true, config: '{}', createdAt: new Date() },
                { name: 'Zendesk', slug: 'zendesk', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Slack', slug: 'slack', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Segment', slug: 'segment', connected: false, config: '{}', createdAt: new Date() },
                { name: 'Intercom', slug: 'intercom', connected: false, config: '{}', createdAt: new Date() },
            ];
            const batch = db.batch();
            integrations.forEach(i => batch.set(db.collection('integrations').doc(), i));
            await batch.commit();
            console.log('[SEED] Integrations seeded to Firestore');
        }

        // Seed default agent if empty
        const agentSnap = await db.collection('agents').limit(1).get();
        if (agentSnap.empty) {
            await db.collection('agents').add({
                name: 'Recharge Assistant',
                description: 'Primary inbound voice agent for DishTV recharge',
                status: 'active',
                systemPrompt: 'You are a helpful DishTV customer service agent. Help customers with their recharge queries.',
                openingLine: 'Hello! This is DishTV support. How can I help you today?',
                voice: 'nova',
                language: 'en-IN',
                llmModel: 'callex-1.3',
                prosodyRate: 1.0,
                prosodyPitch: 1.0,
                patienceMs: 800,
                bargeInMode: 'balanced',
                createdAt: new Date(),
                updatedAt: new Date(),
            });
            console.log('[SEED] Default agent seeded to Firestore');
        }
    } catch (e) {
        console.error('[SEED] Error:', e.message);
    }
}
