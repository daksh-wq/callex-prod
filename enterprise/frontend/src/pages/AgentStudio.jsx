import { useEffect, useState, useRef } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { useAuth } from '../contexts/AuthContext.jsx';
import { Plus, Save, Trash2, Copy, Bot, Mic, Cpu, PhoneCall, ShieldAlert, Sparkles, SlidersHorizontal, Settings, Volume2, Globe, Wrench, FileArchive, X, Send, Loader2, AlertTriangle, Clock, Users } from 'lucide-react';

// AI Noise Suppression (RNNoise deep neural network)
import { RnnoiseWorkletNode, loadRnnoise } from '@sapphi-red/web-noise-suppressor';
import rnnoiseWorkletPath from '@sapphi-red/web-noise-suppressor/rnnoiseWorklet.js?url';
import rnnoiseWasmPath from '@sapphi-red/web-noise-suppressor/rnnoise.wasm?url';

// Callex AI Models
const CALLEX_MODELS = [
    { value: 'callex-1.1', label: 'Callex-1.1', desc: 'Fast · Optimized for high-volume calls' },
    { value: 'callex-1.2', label: 'Callex-1.2', desc: 'Balanced · Best accuracy + speed' },
    { value: 'callex-1.3', label: 'Callex-1.3', desc: 'Advanced · Complex reasoning & RAG' },
];

const TAB_LABELS = [
    { id: 'identity', label: 'Identity & Prompt', icon: Sparkles },
    { id: 'speech', label: 'Speech & Voice', icon: Mic },
    { id: 'telephony', label: 'Telephony & Post-Call', icon: PhoneCall },
    { id: 'analysis', label: 'Custom Analysis', icon: FileArchive }
];

export default function AgentStudio() {
    const { userRole } = useAuth();
    if (userRole === 'superadmin') return <AdminAgentStudio />;
    return <UserAgentStudio />;
}

// ═══════════════════════════════════════════════════════════
// ADMIN AGENT STUDIO — Simplified tuning + maintenance mode
// ═══════════════════════════════════════════════════════════
function AdminAgentStudio() {
    const [userGroups, setUserGroups] = useState([]);
    const [agents, setAgents] = useState([]);
    const [selected, setSelected] = useState(null);
    const [form, setForm] = useState({});
    const [saving, setSaving] = useState(false);
    const [maintenanceModal, setMaintenanceModal] = useState(false);
    const [maintenanceDuration, setMaintenanceDuration] = useState(60);
    const [maintenanceLoading, setMaintenanceLoading] = useState(false);
    const [expandedUser, setExpandedUser] = useState(null);
    const [sidebarMode, setSidebarMode] = useState('by-user'); // 'by-user' or 'all'
    const { showToast } = useStore();

    useEffect(() => { loadData(); }, []);

    async function loadData() {
        try {
            const [grouped, allAgents] = await Promise.all([
                api.get('/admin/agents-by-user'),
                api.get('/admin/agents'),
            ]);
            setUserGroups(grouped);
            setAgents(allAgents);
            // Auto-expand first user with agents
            const first = grouped.find(g => g.agents.length > 0);
            if (first && !expandedUser) setExpandedUser(first.user.id);
        } catch (e) { showToast(e.message, 'error'); }
    }

    function selectAgent(a) {
        setSelected(a);
        setForm({ prosodyRate: a.prosodyRate ?? 1.0, llmModel: a.llmModel || 'callex-1.2', patienceMs: a.patienceMs ?? 800, bargeInMode: a.bargeInMode || 'balanced' });
    }

    async function saveAgent() {
        if (!selected) return;
        setSaving(true);
        try {
            await api.patch(`/admin/agents/${selected.id}`, form);
            showToast('Agent updated', 'success');
            loadData();
        } catch (e) { showToast(e.message, 'error'); }
        finally { setSaving(false); }
    }

    async function activateMaintenance() {
        setMaintenanceLoading(true);
        try {
            await api.post('/admin/maintenance', { durationMinutes: maintenanceDuration });
            showToast(`Maintenance mode activated for ${maintenanceDuration} minutes`, 'success');
            setMaintenanceModal(false);
            loadData();
        } catch (e) { showToast(e.message, 'error'); }
        finally { setMaintenanceLoading(false); }
    }

    const totalAgents = userGroups.reduce((sum, g) => sum + g.totalAgents, 0);
    const usersWithAgents = userGroups.filter(g => g.totalAgents > 0).length;

    return (
        <div className="space-y-6 pb-20">
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Admin Agent Studio</h1>
                    <p className="text-sm text-gray-400">{totalAgents} agents across {usersWithAgents} users</p>
                </div>
                <div className="flex gap-2">
                    <button className="btn-secondary flex items-center gap-2 text-sm" onClick={() => setSidebarMode(m => m === 'by-user' ? 'all' : 'by-user')}>
                        <Users size={14} /> {sidebarMode === 'by-user' ? 'Show All' : 'Group by User'}
                    </button>
                    <button className="btn-primary flex items-center gap-2 bg-amber-500 hover:bg-amber-600" onClick={() => setMaintenanceModal(true)}>
                        <AlertTriangle size={15} /> Maintenance Mode
                    </button>
                </div>
            </div>

            <div className="flex gap-6 h-[calc(100vh-160px)]">
                {/* Agent List Sidebar — User-wise */}
                <div className="w-80 shrink-0 overflow-y-auto pr-2 custom-scroll space-y-1">
                    {sidebarMode === 'by-user' ? (
                        userGroups.map(group => (
                            <div key={group.user.id} className="mb-1">
                                <button
                                    onClick={() => setExpandedUser(expandedUser === group.user.id ? null : group.user.id)}
                                    className={`w-full text-left p-3 rounded-xl border transition-all ${expandedUser === group.user.id ? 'border-orange-200 bg-orange-50/50' : 'border-gray-100 bg-white hover:border-orange-100'}`}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white text-[10px] font-bold">
                                                {group.user.name?.[0]?.toUpperCase() || group.user.email[0].toUpperCase()}
                                            </div>
                                            <div>
                                                <div className="text-xs font-bold text-gray-800 truncate max-w-[160px]">{group.user.name || group.user.email}</div>
                                                <div className="text-[10px] text-gray-400">{group.user.email}</div>
                                            </div>
                                        </div>
                                        <span className="text-[10px] font-bold bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">{group.totalAgents}</span>
                                    </div>
                                </button>
                                {expandedUser === group.user.id && (
                                    <div className="ml-4 mt-1 space-y-1 border-l-2 border-orange-200/50 pl-3">
                                        {group.agents.length === 0 ? (
                                            <p className="text-[10px] text-gray-400 py-2 pl-2">No agents created</p>
                                        ) : (
                                            group.agents.map(a => (
                                                <button key={a.id} onClick={() => selectAgent(a)} className={`w-full text-left p-2.5 rounded-lg border transition-all ${selected?.id === a.id ? 'border-orange-300 bg-orange-50' : 'border-gray-50 bg-white hover:border-orange-100'}`}>
                                                    <div className="flex items-center gap-2">
                                                        <Bot size={12} className="text-orange-500" />
                                                        <span className="font-semibold text-xs text-gray-800 truncate">{a.name}</span>
                                                    </div>
                                                    <div className="flex items-center gap-2 mt-0.5">
                                                        <span className={a.status === 'active' ? 'badge-green' : a.status === 'paused' ? 'badge-orange' : 'badge-gray'} style={{ fontSize: '9px' }}>{a.status}</span>
                                                        <span className="text-[9px] text-gray-400">{a.llmModel}</span>
                                                    </div>
                                                </button>
                                            ))
                                        )}
                                    </div>
                                )}
                            </div>
                        ))
                    ) : (
                        agents.map(a => (
                            <button key={a.id} onClick={() => selectAgent(a)} className={`w-full text-left p-3.5 rounded-xl border transition-all ${selected?.id === a.id ? 'border-orange-200 bg-orange-50' : 'border-gray-100 bg-white hover:border-orange-100'}`}>
                                <div className="flex items-center gap-2">
                                    <Bot size={14} className="text-orange-500" />
                                    <span className="font-semibold text-sm text-gray-800 truncate">{a.name}</span>
                                </div>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className={a.status === 'active' ? 'badge-green' : a.status === 'paused' ? 'badge-orange' : 'badge-gray'} style={{ fontSize: '10px' }}>{a.status}</span>
                                    <span className="text-[10px] text-gray-400">{a.llmModel}</span>
                                </div>
                                {a.user && <div className="text-[10px] text-gray-400 mt-1 flex items-center gap-1"><Users size={10} />{a.user.name || a.user.email}</div>}
                            </button>
                        ))
                    )}
                    {userGroups.length === 0 && <p className="text-xs text-gray-400 text-center py-6">No agents found across any accounts.</p>}
                </div>

                {/* Admin Edit Panel */}
                {selected ? (
                    <div className="flex-1 bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden flex flex-col">
                        <div className="p-5 border-b border-gray-100 bg-gray-50/50 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <Bot size={22} className="text-orange-500" />
                                <div>
                                    <h2 className="text-lg font-bold text-gray-900">{selected.name}</h2>
                                    <p className="text-xs text-gray-400">Owner: {selected.user?.name || selected.user?.email || 'Unknown'} · Status: {selected.status}</p>
                                </div>
                            </div>
                            <button className="btn-primary py-2 text-sm" onClick={saveAgent} disabled={saving}>
                                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}{saving ? ' Saving...' : ' Save Changes'}
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto p-8">
                            <div className="max-w-2xl mx-auto space-y-8">
                                <div className="text-center mb-6">
                                    <h3 className="text-lg font-bold text-gray-800 mb-1">Agent Tuning Parameters</h3>
                                    <p className="text-sm text-gray-400">Adjust core performance settings for this agent</p>
                                </div>

                                {/* Prosody Speed */}
                                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                                    <div className="flex justify-between items-center mb-3">
                                        <div className="flex items-center gap-2">
                                            <Volume2 size={18} className="text-blue-500" />
                                            <span className="font-bold text-gray-800">Prosody Speed</span>
                                        </div>
                                        <span className="text-lg font-black text-orange-600">{(form.prosodyRate ?? 1.0).toFixed(1)}x</span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-3">Controls how fast the agent speaks. Lower = slower, higher = faster.</p>
                                    <input type="range" min="0.5" max="2.0" step="0.1" value={form.prosodyRate ?? 1.0} onChange={e => setForm(f => ({ ...f, prosodyRate: parseFloat(e.target.value) }))} className="w-full accent-orange-500" />
                                    <div className="flex justify-between text-[10px] text-gray-400 mt-1"><span>0.5x (Slow)</span><span>1.0x (Normal)</span><span>2.0x (Fast)</span></div>
                                </div>

                                {/* LLM Model */}
                                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Cpu size={18} className="text-purple-500" />
                                        <span className="font-bold text-gray-800">LLM Model</span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-3">Select the AI model that powers this agent's responses.</p>
                                    <div className="space-y-2">
                                        {CALLEX_MODELS.map(m => (
                                            <label key={m.value} className={`flex items-center gap-3 p-3 rounded-xl border-2 cursor-pointer transition-all ${form.llmModel === m.value ? 'border-orange-300 bg-orange-50' : 'border-gray-100 hover:border-gray-200'}`}>
                                                <input type="radio" name="model" value={m.value} checked={form.llmModel === m.value} onChange={() => setForm(f => ({ ...f, llmModel: m.value }))} className="accent-orange-500" />
                                                <div>
                                                    <div className="text-sm font-semibold text-gray-800">{m.label}</div>
                                                    <div className="text-[10px] text-gray-400">{m.desc}</div>
                                                </div>
                                            </label>
                                        ))}
                                    </div>
                                </div>

                                {/* End-of-Turn Patience */}
                                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                                    <div className="flex justify-between items-center mb-3">
                                        <div className="flex items-center gap-2">
                                            <Clock size={18} className="text-emerald-500" />
                                            <span className="font-bold text-gray-800">End-of-Turn Patience</span>
                                        </div>
                                        <span className="text-lg font-black text-orange-600">{form.patienceMs ?? 800} ms</span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-3">How long to wait for silence before assuming the user has finished speaking.</p>
                                    <input type="range" min="200" max="3000" step="100" value={form.patienceMs ?? 800} onChange={e => setForm(f => ({ ...f, patienceMs: parseInt(e.target.value) }))} className="w-full accent-orange-500" />
                                    <div className="flex justify-between text-[10px] text-gray-400 mt-1"><span>200ms (Aggressive)</span><span>800ms (Default)</span><span>3000ms (Patient)</span></div>
                                </div>

                                {/* Barge-in Mode */}
                                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Mic size={18} className="text-red-500" />
                                        <span className="font-bold text-gray-800">Barge-in / Interruption Mode</span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-3">Controls how the agent handles user interruptions while speaking.</p>
                                    <select className="input-field text-sm" value={form.bargeInMode || 'balanced'} onChange={e => setForm(f => ({ ...f, bargeInMode: e.target.value }))}>
                                        <option value="disabled">Disabled — Do not interrupt bot</option>
                                        <option value="polite">Polite — Only interrupt on full sentences</option>
                                        <option value="balanced">Balanced — Interrupt on 3+ words</option>
                                        <option value="aggressive">Aggressive — Interrupt instantly on noise</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-gray-400 bg-gray-50/30 rounded-2xl border border-gray-100 border-dashed">
                        <div className="text-center"><Bot size={48} className="mx-auto mb-3 opacity-20 text-orange-500" /><p className="text-sm font-semibold text-gray-400">Select an agent to view and tune its parameters</p></div>
                    </div>
                )}
            </div>

            {/* Maintenance Mode Modal */}
            {maintenanceModal && (
                <div className="fixed inset-0 z-50 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center">
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-5">
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2"><AlertTriangle size={20} className="text-amber-500" /> Maintenance Mode</h3>
                            <button onClick={() => setMaintenanceModal(false)} className="p-1 hover:bg-gray-100 rounded-lg"><X size={18} /></button>
                        </div>
                        <div className="bg-amber-50 border border-amber-100 p-4 rounded-xl">
                            <p className="text-sm text-amber-800"><strong>Warning:</strong> This will pause ALL active agents across the entire platform and send a notification to all users.</p>
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-gray-600 mb-2 block">Maintenance Duration</label>
                            <div className="flex gap-2">
                                {[30, 60, 120].map(mins => (
                                    <button key={mins} onClick={() => setMaintenanceDuration(mins)} className={`flex-1 py-3 rounded-xl text-sm font-bold transition-all ${maintenanceDuration === mins ? 'bg-amber-500 text-white shadow-lg' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
                                        {mins} min
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="flex justify-end gap-2 pt-2">
                            <button onClick={() => setMaintenanceModal(false)} className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700">Cancel</button>
                            <button onClick={activateMaintenance} disabled={maintenanceLoading} className="btn-primary bg-amber-500 hover:bg-amber-600 flex items-center gap-2">
                                {maintenanceLoading ? <Loader2 size={14} className="animate-spin" /> : <AlertTriangle size={14} />}
                                {maintenanceLoading ? 'Activating...' : 'Activate Maintenance'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ═══════════════════════════════════════════════════════════
// USER AGENT STUDIO — Full-featured agent creation & editing
// ═══════════════════════════════════════════════════════════
function UserAgentStudio() {
    const [agents, setAgents] = useState([]);
    const [selected, setSelected] = useState(null);
    const [form, setForm] = useState({});
    const [tab, setTab] = useState('identity');
    const [promptVersions, setPromptVersions] = useState([]);
    const [newAgent, setNewAgent] = useState(false);
    const [showSimulation, setShowSimulation] = useState(false);
    const [listTab, setListTab] = useState('production');
    const [pushingProd, setPushingProd] = useState(false);
    const { showToast } = useStore();

    const fetchAgents = () => api.agents().then(res => setAgents(Array.isArray(res) ? res : (res?.agents || [])));
    const fetchVersions = (id) => api.agentPromptVersions(id).then(setPromptVersions);

    useEffect(() => { fetchAgents(); }, []);

    function selectAgent(a) {
        setSelected(a); setNewAgent(false);
        setForm({
            ...a,
            fillerPhrases: tryParse(a.fillerPhrases, []),
            ipaLexicon: tryParse(a.ipaLexicon, {}),
            tools: tryParse(a.tools, []),
            analysisSchema: tryParse(a.analysisSchema, []),
        });
        fetchVersions(a.id);
    }

    function tryParse(val, def) { try { return JSON.parse(val); } catch { return def; } }

    async function saveAgent() {
        try {
            const payload = { ...form, fillerPhrases: form.fillerPhrases, ipaLexicon: form.ipaLexicon, tools: form.tools, analysisSchema: JSON.stringify(form.analysisSchema || []) };
            if (newAgent) {
                const a = await api.createAgent(payload);
                showToast('Agent created', 'success'); fetchAgents(); selectAgent(a);
            } else {
                const a = await api.updateAgent(selected.id, payload);
                showToast('Agent saved', 'success'); fetchAgents(); setSelected(a);
            }
        } catch (e) {
            showToast(e.message || 'Failed to save agent', 'error');
        }
    }

    async function deleteAgent(id) {
        if (!window.confirm("Delete this agent?")) return;
        try {
            await api.deleteAgent(id);
            showToast('Agent deleted', 'info'); setSelected(null); setForm({}); fetchAgents();
        } catch (e) {
            showToast(e.message || 'Failed to delete agent', 'error');
        }
    }

    async function pushToProd(id) {
        if (!window.confirm("Are you sure? This will permanently overwrite the Production Agent's system prompt with this Sandbox's prompt!")) return;
        setPushingProd(true);
        try {
            await api.post(`/agents/${id}/push-to-prod`);
            showToast('Successfully pushed Sandbox to Production!', 'success');
            // Refresh to show updated parent agent if selected
            fetchAgents();
        } catch (e) {
            showToast(e.message || 'Failed to push to production', 'error');
        } finally {
            setPushingProd(false);
        }
    }

    async function setStatus(id, status) {
        await api.setAgentStatus(id, status);
        showToast(`Agent ${status}`, 'success'); fetchAgents();
    }

    async function savePromptVersion() {
        if (!selected) return;
        await api.savePromptVersion(selected.id, { prompt: form.systemPrompt, label: `v${promptVersions.length + 1} — Manual save` });
        showToast('Prompt version saved', 'success'); fetchVersions(selected.id);
    }

    const F = (key) => ({ value: form[key] || '', onChange: e => setForm(f => ({ ...f, [key]: e.target.value })) });
    const FNum = (key) => ({ value: form[key] ?? '', onChange: e => setForm(f => ({ ...f, [key]: parseFloat(e.target.value) })) });
    const FBool = (key) => ({ checked: form[key] ?? false, onChange: e => setForm(f => ({ ...f, [key]: e.target.checked })) });

    return (
        <div className="space-y-6 pb-20">
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Advanced Agent Studio</h1>
                    <p className="text-sm text-gray-400">Configure massive 15+ detailed voice, RAG, and telephony parameters</p>
                </div>
                <button className="btn-primary" onClick={() => { setNewAgent(true); setSelected(null); setForm({ name: '', status: 'draft', sttEngine: 'callex-1.1', llmModel: 'callex-1.3', bargeInMode: 'balanced' }); setTab('identity'); }}>
                    <Plus size={15} /> New Agent
                </button>
            </div>

            <div className="flex gap-6 h-[calc(100vh-160px)]">
                {/* Agent List Sidebar */}
                <div className="w-72 shrink-0 bg-white rounded-2xl border border-gray-100 shadow-sm flex flex-col overflow-hidden">
                    <div className="flex border-b border-gray-100 p-2 gap-1 bg-gray-50/50">
                        <button onClick={() => setListTab('production')} className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-all ${listTab === 'production' ? 'bg-white shadow-sm text-gray-900 border border-gray-150' : 'text-gray-500 hover:text-gray-700'}`}>Production</button>
                        <button onClick={() => setListTab('sandbox')} className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-all ${listTab === 'sandbox' ? 'bg-white shadow-sm text-gray-900 border border-gray-150' : 'text-gray-500 hover:text-gray-700'}`}>Sandbox</button>
                    </div>
                    <div className="flex-1 space-y-2 overflow-y-auto p-3 custom-scroll">
                        {(listTab === 'production' ? agents.filter(a => !a.isTrainingSandbox) : agents.filter(a => a.isTrainingSandbox)).map(a => (
                            <button key={a.id} onClick={() => selectAgent(a)} className={`w-full text-left p-3.5 rounded-xl border transition-all ${selected?.id === a.id ? 'border-orange-200 bg-orange-50' : 'border-gray-100 bg-white hover:border-orange-100'}`}>
                                <div className="flex items-center gap-2">
                                    <Bot size={14} className="text-orange-500" />
                                    <span className="font-semibold text-sm text-gray-800 truncate">{a.name}</span>
                                </div>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className={a.status === 'active' ? 'badge-green' : a.status === 'paused' ? 'badge-orange' : 'badge-gray'} style={{ fontSize: '10px' }}>{a.status}</span>
                                    {a.isTrainingSandbox && <span className="badge-purple" style={{ fontSize: '10px' }}>Auto-Learning</span>}
                                </div>
                            </button>
                        ))}
                        {(listTab === 'production' ? agents.filter(a => !a.isTrainingSandbox) : agents.filter(a => a.isTrainingSandbox)).length === 0 && (
                            <p className="text-xs text-gray-400 text-center py-6">
                                {listTab === 'production' ? 'No production agents yet.' : 'No sandbox agents created.'}
                            </p>
                        )}
                    </div>
                </div>

                {/* Main Configuration Panel */}
                {(selected || newAgent) ? (
                    <div className="flex-1 flex flex-col bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">

                        {/* Top Sticky Header for Agent Name & Save */}
                        <div className="p-5 border-b border-gray-100 bg-gray-50/50 flex items-center justify-between sticky top-0 z-10">
                            <div className="flex items-center gap-4 flex-1">
                                <input className="text-xl font-bold text-gray-900 bg-transparent border-none outline-none placeholder-gray-300 w-1/3" placeholder="Agent Name..." {...F('name')} />
                                <select className="text-xs font-semibold bg-white border border-gray-200 rounded-lg px-2 py-1 outline-none text-gray-600" value={form.status || 'draft'} onChange={e => setForm(f => ({ ...f, status: e.target.value }))}>
                                    <option value="draft">Draft (Testing)</option><option value="active">Active (Production)</option><option value="paused">Paused</option>
                                </select>
                            </div>
                            <div className="flex gap-2">
                                {selected?.isTrainingSandbox && (
                                    <button className="btn-primary py-2 text-sm bg-purple-600 hover:bg-purple-700 shadow-purple-200" onClick={() => pushToProd(selected.id)} disabled={pushingProd}>
                                        {pushingProd ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />} {pushingProd ? 'Pushing...' : 'Push to Production'}
                                    </button>
                                )}
                                {selected && <button className="btn-secondary py-2 text-blue-600 border-blue-100 hover:bg-blue-50" onClick={() => setShowSimulation(true)}><PhoneCall size={14} /> Simulate</button>}
                                <button className="btn-primary py-2 text-sm" onClick={saveAgent}><Save size={14} /> Save</button>
                                {selected && <button className="btn-secondary py-2 text-red-600 border-red-100 hover:bg-red-50" onClick={() => deleteAgent(selected.id)}><Trash2 size={14} /></button>}
                            </div>
                        </div>

                        <div className="flex flex-1 overflow-hidden">
                            {/* Inner Vertical Tabs */}
                            <div className="w-56 bg-white border-r border-gray-100 flex flex-col gap-1 p-3 overflow-y-auto">
                                {TAB_LABELS.map(t => {
                                    const Icon = t.icon;
                                    const isActive = tab === t.id;
                                    return (
                                        <button key={t.id} onClick={() => setTab(t.id)} className={`flex items-center gap-3 px-3 py-3 rounded-xl text-xs font-bold transition-all ${isActive ? 'bg-orange-50 text-orange-700' : 'text-gray-500 hover:bg-gray-50'}`}>
                                            <Icon size={16} className={isActive ? 'text-orange-500' : 'text-gray-400'} />
                                            {t.label}
                                        </button>
                                    );
                                })}
                            </div>

                            {/* Scrollable Form Content */}
                            <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scroll bg-gray-50/30">

                                {/* TAB: IDENTITY & PROMPT */}
                                {tab === 'identity' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="grid grid-cols-2 gap-6">
                                            <div><label className="label text-gray-700">Agent Description</label><input className="input-field max-w-2xl text-sm" placeholder="e.g. Inbound billing support bot for North America..." {...F('description')} /></div>
                                            <div>
                                                <label className="label text-gray-700">Agent Language</label>
                                                <select className="input-field text-sm" value={form.language || 'en-US'} onChange={e => setForm(f => ({ ...f, language: e.target.value }))}>
                                                    <option value="en-US">English (US)</option>
                                                    <option value="en-GB">English (UK)</option>
                                                    <option value="es-ES">Spanish</option>
                                                    <option value="fr-FR">French</option>
                                                    <option value="de-DE">German</option>
                                                    <option value="hi-IN">Hindi</option>
                                                    <option value="gu-IN">Gujarati</option>
                                                </select>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-6">
                                            <div>
                                                <label className="label">TTS Opening Line</label>
                                                <textarea className="input-field h-20 text-sm" placeholder="Hello! I am Callex, how can I help you today?" {...F('openingLine')} />
                                            </div>
                                            <div>
                                                <label className="label">RAG Fallback Message</label>
                                                <textarea className="input-field h-20 text-sm" placeholder="I'm sorry, I couldn't find that in my knowledge base. Let me transfer you." {...F('fallbackMessage')} />
                                                <p className="text-[10px] text-gray-400 mt-1">Played when documentation similarity threshold is missed.</p>
                                            </div>
                                        </div>

                                        <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4 shadow-sm">
                                            <div className="flex items-center justify-between">
                                                <h3 className="font-bold text-gray-800 flex items-center gap-2"><Sparkles size={16} className="text-blue-500" /> Train your Script ("System Prompt")</h3>
                                                <button className="btn-secondary text-xs py-1 px-3" onClick={savePromptVersion}><Copy size={12} /> Save Revision</button>
                                            </div>
                                            <textarea className="input-field h-64 font-mono text-xs leading-5 p-4 bg-gray-50" placeholder="You are a helpful assistant..." {...F('systemPrompt')} />
                                        </div>

                                        <div className="grid grid-cols-2 gap-6">
                                            <div className="bg-white p-4 rounded-xl border border-gray-100">
                                                <label className="label mb-2 flex items-center gap-2"><ShieldAlert size={14} className="text-red-400" /> Profanity Filter</label>
                                                <select className="input-field text-sm" value={form.profanityFilter || 'redact'} onChange={e => setForm(f => ({ ...f, profanityFilter: e.target.value }))}>
                                                    <option value="redact">Redact (Bleep out profanity)</option>
                                                    <option value="block">Block (Hang up immediately)</option>
                                                    <option value="allow">Allow (No filtering)</option>
                                                </select>
                                            </div>
                                            <div className="bg-white p-4 rounded-xl border border-gray-100 flex items-center justify-between">
                                                <div>
                                                    <label className="label mb-0">Strict Topic Restriction</label>
                                                    <p className="text-xs text-gray-400 mt-1">Prevent LLM from going off-topic</p>
                                                </div>
                                                <label className="relative inline-flex items-center cursor-pointer">
                                                    <input type="checkbox" className="sr-only peer" {...FBool('topicRestriction')} />
                                                    <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-orange-500"></div>
                                                </label>
                                            </div>
                                        </div>

                                        <div className="bg-white p-4 rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50 to-white shadow-sm flex items-center justify-between">
                                            <div>
                                                <label className="label mb-0 flex items-center gap-2"><Sparkles size={14} className="text-blue-600" /> Advanced NLP & Persuasion Mode</label>
                                                <p className="text-xs text-blue-800 mt-1">Acts as a highly persuasive human with advanced EQ to multiply conversation time and handle objections.</p>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer">
                                                <input type="checkbox" className="sr-only peer" {...FBool('advancedNlpEnabled')} />
                                                <div className="w-9 h-5 bg-blue-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-blue-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                                            </label>
                                        </div>
                                    </div>
                                )}

                                {/* TAB: SPEECH & VOICE */}
                                {tab === 'speech' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="grid grid-cols-2 gap-6">
                                            <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4">
                                                <h3 className="font-bold text-gray-800 border-b border-gray-50 pb-2 flex items-center gap-2"><Volume2 size={16} className="text-blue-500" /> Voice Synthesis (TTS)</h3>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Voice Persona</label>
                                                    <div className="flex flex-col gap-2">
                                                        <div className="flex items-center gap-2">
                                                            <select className="input-field text-sm flex-1" value={form.voice || 'MF4J4IDTRo0AxOO4dpFR'} onChange={e => setForm(f => ({ ...f, voice: e.target.value }))}>
                                                                <option value="MF4J4IDTRo0AxOO4dpFR">Devi (Clear Hindi)</option>
                                                                <option value="1qEiC6qsybMkmnNdVMbK">Monika (Modulated, Professional)</option>
                                                                <option value="qDuRKMlYmrm8trt5QyBn">Taksh (Powerful & Commanding)</option>
                                                                <option value="LQ2auZHpAQ9h4azztqMT">Parveen (Confident Male)</option>
                                                                <option value="s6cZdgI3j07hf4frz4Q8">Arvi (Desi Conversational)</option>
                                                                <optgroup label="US English Voices">
                                                                    <option value="21m00Tcm4TlvDq8ikWAM">Rachel (US Professional Female)</option>
                                                                    <option value="EXAVITQu4vr4xnSDxMaL">Bella (US Friendly Female)</option>
                                                                    <option value="CYw3kZ02Hs0563khs1Fj">Dave (US Conversational Male)</option>
                                                                    <option value="ErXwobaYiN019PkySvjV">Antoni (US Well-Rounded Male)</option>
                                                                </optgroup>
                                                                {form.clonedVoiceId && <option value={form.clonedVoiceId}>Custom Cloned Voice ({form.clonedVoiceId})</option>}
                                                            </select>
                                                            <button type="button" className={`btn-secondary py-2 px-3 flex items-center gap-2 ${form.previewLoading ? 'opacity-50 cursor-not-allowed' : ''}`} disabled={form.previewLoading} title="Play 5s Hindi Preview" onClick={async () => {
                                                                try {
                                                                    setForm(f => ({ ...f, previewLoading: true }));
                                                                    // Use the new dynamic streaming endpoint based on current prosody configurations
                                                                    const res = await fetch('http://localhost:4000/api/agents/tts-preview', {
                                                                        method: 'POST',
                                                                        headers: { 'Content-Type': 'application/json' },
                                                                        body: JSON.stringify({
                                                                            voiceId: form.voice || 'MF4J4IDTRo0AxOO4dpFR',
                                                                            prosodyRate: form.prosodyRate ?? 1.0,
                                                                            prosodyPitch: form.prosodyPitch ?? 1.0
                                                                        })
                                                                    });

                                                                    if (!res.ok) throw new Error('API failed');
                                                                    const blob = await res.blob();
                                                                    const url = URL.createObjectURL(blob);
                                                                    const audio = new Audio(url);
                                                                    audio.onended = () => URL.revokeObjectURL(url);
                                                                    await audio.play();
                                                                } catch (e) {
                                                                    console.error("Audio play failed:", e);
                                                                    showToast("Failed to fetch dynamic preview", "error");
                                                                } finally {
                                                                    setForm(f => ({ ...f, previewLoading: false }));
                                                                }
                                                            }}>
                                                                <Volume2 size={14} className={form.previewLoading ? "text-gray-400 animate-pulse" : "text-orange-500"} />
                                                                <span className="text-xs">{form.previewLoading ? "Loading..." : "Preview"}</span>
                                                            </button>
                                                        </div>
                                                        <label className={`btn-primary py-2 px-3 flex items-center justify-center gap-2 text-xs cursor-pointer w-full transition-all ${form.cloningStatus ? 'opacity-50 pointer-events-none' : ''}`}>
                                                            <Mic size={14} className={form.cloningStatus ? 'animate-pulse' : ''} />
                                                            <span>{form.cloningStatus || "🎙️ Train Own Voice (Upload MP3/WAV)"}</span>
                                                            <input type="file" accept="audio/*" className="hidden" onChange={async (e) => {
                                                                const file = e.target.files[0];
                                                                if (!file) return;
                                                                setForm(f => ({ ...f, cloningStatus: "Uploading & Cloning... Please wait." }));
                                                                const formData = new FormData();
                                                                formData.append('audio', file);
                                                                try {
                                                                    const res = await api.cloneVoice(formData);
                                                                    setForm(f => ({ ...f, voice: res.voiceId, clonedVoiceId: res.voiceId, cloningStatus: null }));
                                                                    showToast("Voice cloned and applied successfully!", "success");
                                                                } catch (err) {
                                                                    console.error(err);
                                                                    showToast("Failed to clone voice", "error");
                                                                    setForm(f => ({ ...f, cloningStatus: null }));
                                                                }
                                                                e.target.value = null; // reset input
                                                            }} />
                                                        </label>
                                                    </div>
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Speaking Style</label>
                                                    <select className="input-field text-sm" value={form.speakingStyle || 'professional'} onChange={e => setForm(f => ({ ...f, speakingStyle: e.target.value }))}>
                                                        <option value="professional">Professional / Corporate</option><option value="friendly">Friendly / Casual</option><option value="urgent">Urgent / Fast-paced</option><option value="empathetic">Empathetic / Soft</option>
                                                    </select>
                                                </div>
                                                <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                                                    <div className="flex justify-between text-[11px] font-bold text-gray-500 mb-2 uppercase"><span>Prosody Speed</span><span className="text-orange-600">{form.prosodyRate ?? 1.0}x</span></div>
                                                    <input type="range" min="0.5" max="2.0" step="0.1" value={form.prosodyRate ?? 1.0} onChange={e => setForm(f => ({ ...f, prosodyRate: parseFloat(e.target.value) }))} className="w-full accent-orange-500 mb-4" />
                                                    <div className="flex justify-between text-[11px] font-bold text-gray-500 mb-2 uppercase"><span>Pitch Modulation</span><span className="text-orange-600">{(form.prosodyPitch ?? 1.0).toFixed(1)}</span></div>
                                                    <input type="range" min="0.5" max="2.0" step="0.1" value={form.prosodyPitch ?? 1.0} onChange={e => setForm(f => ({ ...f, prosodyPitch: parseFloat(e.target.value) }))} className="w-full accent-orange-500" />
                                                </div>
                                            </div>

                                            <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4">
                                                <h3 className="font-bold text-gray-800 border-b border-gray-50 pb-2 flex items-center gap-2"><Mic size={16} className="text-green-500" /> Speech Recognition (STT)</h3>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Voice Engine</label>
                                                    <select className="input-field text-sm" value={form.sttEngine || 'callex-1.1'} onChange={e => setForm(f => ({ ...f, sttEngine: e.target.value }))}>
                                                        <option value="callex-1.1">Callex-1.1 (General Accents)</option><option value="callex-1.2">Callex-1.2 (Thick Accents / Code-switching)</option><option value="callex-1.3">Callex-1.3 (Global Multilingual)</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Barge-in / Interruption Mode</label>
                                                    <select className="input-field text-sm" value={form.bargeInMode || 'balanced'} onChange={e => setForm(f => ({ ...f, bargeInMode: e.target.value }))}>
                                                        <option value="disabled">Disabled (Do not interrupt bot)</option><option value="polite">Polite (Only interrupt on full sentences)</option><option value="balanced">Balanced (Interrupt on 3+ words)</option><option value="aggressive">Aggressive (Interrupt instantly on noise)</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 flex justify-between"><span>End-of-Turn Patience</span><span className="text-orange-600">{form.patienceMs ?? 800} ms</span></label>
                                                    <p className="text-[10px] text-gray-400 mb-2">How long to wait for silence before assuming user is done speaking.</p>
                                                    <input type="range" min="200" max="3000" step="100" value={form.patienceMs ?? 800} onChange={e => setForm(f => ({ ...f, patienceMs: parseInt(e.target.value) }))} className="w-full accent-orange-500" />
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-white p-5 rounded-xl border border-gray-100">
                                            <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2"><Globe size={16} className="text-purple-500" /> Environmental Acoustics</h3>
                                            <div className="grid grid-cols-2 gap-6">
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Background Ambience Injection</label>
                                                    <p className="text-[10px] text-gray-400 mb-2">Inject synthetic background noise to mask AI silence and sound like a real human.</p>
                                                    <select className="input-field text-sm" value={form.backgroundAmbience || 'none'} onChange={e => setForm(f => ({ ...f, backgroundAmbience: e.target.value }))}>
                                                        <option value="none">None (Crystal Clear)</option><option value="office">Quiet Office (Keyboard typing)</option><option value="call_center">Busy Call Center (Muffled chatter)</option><option value="static">Subtle Analog Static</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Dynamic Cognitive Fillers</label>
                                                    <p className="text-[10px] text-gray-400 mb-2">Bot will say these randomly if API latency exceeds 1.5s.</p>
                                                    <div className="flex flex-wrap gap-1.5 mb-2">
                                                        {(Array.isArray(form.fillerPhrases) ? form.fillerPhrases : []).map((p, i) => (
                                                            <span key={i} className="text-xs bg-orange-50 text-orange-700 px-2 py-1 rounded-md border border-orange-100 cursor-pointer hover:bg-red-50 hover:text-red-700 hover:border-red-200 transition-colors" onClick={() => setForm(f => ({ ...f, fillerPhrases: f.fillerPhrases.filter((_, j) => j !== i) }))}>{p} <span className="text-red-400 ml-1">×</span></span>
                                                        ))}
                                                    </div>
                                                    <input className="input-field text-xs bg-gray-50" placeholder="Type a filler phrase and press Enter... (e.g. 'Just a sec')" onKeyDown={e => { if (e.key === 'Enter' && e.target.value.trim()) { setForm(f => ({ ...f, fillerPhrases: [...(Array.isArray(f.fillerPhrases) ? f.fillerPhrases : []), e.target.value.trim()] })); e.target.value = ''; } }} />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* TAB: TELEPHONY & POST-CALL */}
                                {tab === 'telephony' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="grid grid-cols-2 gap-6">
                                            <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4">
                                                <h3 className="font-bold text-gray-800 mb-2 flex items-center gap-2"><SlidersHorizontal size={16} className="text-gray-800" /> Session & Telephony</h3>

                                                <div className="flex bg-gray-100 p-1 rounded-xl mb-4">
                                                    <button type="button" onClick={() => setForm(f => ({ ...f, callDirection: 'inbound' }))} className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-colors ${(form.callDirection || 'inbound') === 'inbound' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>Inbound Calls</button>
                                                    <button type="button" onClick={() => setForm(f => ({ ...f, callDirection: 'outbound' }))} className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-colors ${form.callDirection === 'outbound' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>Outbound Dialer</button>
                                                </div>

                                                {form.callDirection === 'outbound' && (
                                                    <div className="mb-4 bg-orange-50 border border-orange-100 p-3 rounded-xl flex items-center justify-between">
                                                        <div>
                                                            <div className="text-xs font-bold text-orange-900">Upload Numbers</div>
                                                            <div className="text-[10px] text-orange-700">CSV of contacts to call</div>
                                                        </div>
                                                        <label className="btn-primary py-1.5 px-3 text-xs cursor-pointer">
                                                            <input type="file" accept=".csv" className="hidden" onChange={(e) => { e.target.value = null; showToast('CSV Uploaded! Ready for dialer.', 'success'); }} />
                                                            Upload CSV
                                                        </label>
                                                    </div>
                                                )}

                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Maximum Call Duration Limit (Minutes)</label>
                                                    <input type="number" className="input-field text-sm" {...FNum('maxDuration')} placeholder="30" />
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">No Answer Ring Timeout (Seconds)</label>
                                                    <input type="number" className="input-field text-sm" {...FNum('ringTimeout')} placeholder="30" />
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Voicemail Detection Protocol</label>
                                                    <select className="input-field text-sm" value={form.voicemailLogic || 'hangup'} onChange={e => setForm(f => ({ ...f, voicemailLogic: e.target.value }))}>
                                                        <option value="hangup">Instantly Hang up</option><option value="leave_message">Leave dynamic voice message</option><option value="human_escalate">Wait for human operator</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Call Forwarding Enable</label>
                                                    <select className="input-field text-sm" value={form.callForwarding || 'none'} onChange={e => setForm(f => ({ ...f, callForwarding: e.target.value }))}>
                                                        <option value="none">Disabled</option><option value="human">Forward to Human Operator</option><option value="agent">Forward to Another Agent</option>
                                                    </select>
                                                </div>
                                                <div className="flex items-center justify-between pt-2 border-t border-gray-50">
                                                    <span className="text-xs font-semibold text-gray-700">Listen for DTMF Tones (Dialpad)</span>
                                                    <label className="relative inline-flex items-center cursor-pointer">
                                                        <input type="checkbox" className="sr-only peer" {...FBool('processDtmf')} />
                                                        <div className="w-8 h-4 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-emerald-500"></div>
                                                    </label>
                                                </div>
                                            </div>

                                            <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4">
                                                <h3 className="font-bold text-gray-800 mb-2 flex items-center gap-2"><FileArchive size={16} className="text-emerald-500" /> Pipeline Integrations</h3>
                                                <div>
                                                    <label className="text-xs font-semibold text-gray-500 mb-1 block">Completion Webhook URL</label>
                                                    <p className="text-[10px] text-gray-400 mb-1">Send a POST request immediately when call finishes.</p>
                                                    <input type="text" className="input-field text-sm font-mono" placeholder="https://api.yourcrm.com/webhooks/callex" {...F('webhookUrl')} />
                                                </div>

                                                <div className="space-y-3 pt-4 border-t border-gray-50">
                                                    <div className="flex items-center justify-between bg-gray-50/50 p-2 rounded-lg">
                                                        <span className="text-xs font-semibold text-gray-700">Auto-Generate LLM Summary</span>
                                                        <input type="checkbox" className="w-4 h-4 accent-orange-500" {...FBool('autoSummary')} />
                                                    </div>
                                                    <div className="flex items-center justify-between bg-gray-50/50 p-2 rounded-lg">
                                                        <span className="text-xs font-semibold text-gray-700">Auto-Tag Sentiment Score</span>
                                                        <input type="checkbox" className="w-4 h-4 accent-orange-500" {...FBool('autoSentiment')} />
                                                    </div>
                                                    <div className="flex items-center justify-between bg-gray-50/50 p-2 rounded-lg">
                                                        <span className="text-xs font-semibold text-gray-700">Save MP3 Audio Recording</span>
                                                        <input type="checkbox" className="w-4 h-4 accent-orange-500" {...FBool('recordCall')} />
                                                    </div>
                                                    <div className="flex items-center justify-between bg-gray-50/50 p-2 rounded-lg">
                                                        <div>
                                                            <div className="text-xs font-semibold text-gray-700">Auto Follow-Up Module</div>
                                                            <div className="text-[10px] text-gray-400">Agent schedules texts/emails based on convo</div>
                                                        </div>
                                                        <label className="relative inline-flex items-center cursor-pointer">
                                                            <input type="checkbox" className="sr-only peer" {...FBool('autoFollowUp')} />
                                                            <div className="w-8 h-4 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-orange-500"></div>
                                                        </label>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* TAB: CUSTOM ANALYSIS */}
                                {tab === 'analysis' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="bg-white p-5 rounded-xl border border-gray-100 space-y-4 shadow-sm">
                                            <div>
                                                <h3 className="font-bold text-gray-800 flex items-center gap-2"><FileArchive size={16} className="text-orange-500" /> Custom Call Analysis Params</h3>
                                                <p className="text-sm text-gray-500 mt-1">Define exactly what data the AI should extract from every call and return as structured data.</p>
                                            </div>

                                            <div className="space-y-3">
                                                {(Array.isArray(form.analysisSchema) ? form.analysisSchema : []).map((field, idx) => (
                                                    <div key={idx} className="flex gap-3 items-start bg-gray-50 p-3 rounded-xl border border-gray-100">
                                                        <div className="flex-1 space-y-2">
                                                            <input className="input-field text-sm font-semibold text-gray-800 bg-white" placeholder="Variable Name (e.g., budget)" value={field.name} onChange={e => {
                                                                const newCols = [...(form.analysisSchema || [])];
                                                                newCols[idx] = { ...newCols[idx], name: e.target.value };
                                                                setForm(f => ({ ...f, analysisSchema: newCols }));
                                                            }} />
                                                            <input className="input-field text-xs text-gray-600 bg-white" placeholder="Description/Hint for AI" value={field.description} onChange={e => {
                                                                const newCols = [...(form.analysisSchema || [])];
                                                                newCols[idx] = { ...newCols[idx], description: e.target.value };
                                                                setForm(f => ({ ...f, analysisSchema: newCols }));
                                                            }} />
                                                        </div>
                                                        <div className="w-32">
                                                            <select className="input-field text-xs bg-white" value={field.type} onChange={e => {
                                                                const newCols = [...(form.analysisSchema || [])];
                                                                newCols[idx] = { ...newCols[idx], type: e.target.value };
                                                                setForm(f => ({ ...f, analysisSchema: newCols }));
                                                            }}>
                                                                <option value="string">Text / String</option>
                                                                <option value="number">Number</option>
                                                                <option value="boolean">True/False</option>
                                                            </select>
                                                        </div>
                                                        <button className="btn-secondary py-2 px-3 text-red-500 hover:bg-red-50" onClick={() => {
                                                            setForm(f => ({ ...f, analysisSchema: f.analysisSchema.filter((_, i) => i !== idx) }))
                                                        }}><Trash2 size={14} /></button>
                                                    </div>
                                                ))}
                                                
                                                <button className="btn-secondary w-full border-dashed border-2 text-orange-600 hover:bg-orange-50 py-3 font-semibold text-sm flex items-center justify-center gap-2" onClick={() => {
                                                    setForm(f => ({ ...f, analysisSchema: [...(Array.isArray(f.analysisSchema) ? f.analysisSchema : []), { name: '', description: '', type: 'string' }] }));
                                                }}>
                                                    <Plus size={16} /> Add Extraction Variable
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-gray-400 bg-gray-50/30 rounded-2xl border border-gray-100 border-dashed">
                        <div className="text-center"><Bot size={48} className="mx-auto mb-3 opacity-20 text-orange-500" /><p className="text-sm font-semibold text-gray-400">Select an agent or create a new one to unlock advanced settings</p></div>
                    </div>
                )}
            </div>
            {showSimulation && selected && <LiveSimulationModal agent={form} onClose={() => setShowSimulation(false)} />}
        </div>
    );
}

function LiveSimulationModal({ agent, onClose }) {
    const [history, setHistory] = useState([]);
    const historyRef = useRef([]);
    const [callStatus, setCallStatus] = useState('connecting'); // connecting, listening, ai-speaking, error
    const [duration, setDuration] = useState(0);
    const recognitionRef = useRef(null);
    const speakerActiveRef = useRef(false);
    const isModalOpenRef = useRef(true);
    const audioRef = useRef(null);
    const activeReqIdRef = useRef(0);
    const silenceTimerRef = useRef(null);
    const aiCurrentSentenceRef = useRef("");


    // Call Timer
    useEffect(() => {
        if (callStatus === 'connecting' || callStatus === 'error') return;
        const interval = setInterval(() => setDuration(d => d + 1), 1000);
        return () => clearInterval(interval);
    }, [callStatus]);

    // Background Ambience
    useEffect(() => {
        if (!agent.backgroundAmbience || agent.backgroundAmbience === 'none') return;
        
        const AMBIENCE_URLS = {
            'office': 'https://actions.google.com/sounds/v1/water/rain_on_roof.ogg', 
            'call_center': 'https://actions.google.com/sounds/v1/crowds/restaurant_chatter.ogg',
            'static': 'https://actions.google.com/sounds/v1/alarms/white_noise.ogg'
        };
        
        const src = AMBIENCE_URLS[agent.backgroundAmbience];
        if (!src) return;

        const bgAudio = new Audio(src);
        bgAudio.loop = true;
        bgAudio.volume = 0.05; // 5% volume - faint but realistic
        
        const playPromise = bgAudio.play();
        if (playPromise !== undefined) {
            playPromise.catch(e => console.warn("Background audio blocked by autoplay rules:", e));
        }

        return () => {
            bgAudio.pause();
            bgAudio.src = ""; // Clean up memory
        };
    }, [agent.backgroundAmbience]);

    const formatTime = (secs) => `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`;

    const triggerListen = () => {
        if (!isModalOpenRef.current) return;
        try {
            recognitionRef.current?.start();
            if (!speakerActiveRef.current) {
                setCallStatus('listening');
            }
        } catch (e) { /* already started */ }
    }

    // Initialize CX-STT Voice Engine via AudioWorklet + WebSocket Pipeline
    useEffect(() => {
        let micStream = null;
        let audioContext = null;
        let sttWs = null;
        let predictiveTimer = null;
        let lastPredictedText = '';
        let activePredictionId = null;
        let finalTranscript = '';
        let interimTranscript = '';

        const initCxSTT = async () => {
            try {
                // Step 1: Get microphone with hardware DSP noise suppression
                micStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        noiseSuppression: true,
                        echoCancellation: true,
                        autoGainControl: true,
                        sampleRate: { ideal: 16000 }
                    }
                });
                console.log("🎤 Microphone acquired with Hardware DSP.");

                // Step 2: Set up AudioContext at native 48kHz (needed for RNNoise AI)
                // RNNoise neural network requires 48kHz input for its spectral analysis
                audioContext = new (window.AudioContext || window.webkitAudioContext)();

                // Step 2.5: AI NEURAL NOISE SUPPRESSION (RNNoise Deep RNN)
                // This is the same AI noise cancellation used by Jitsi Meet, Discord, etc.
                // It analyzes spectral patterns to distinguish human voice from ALL background noise
                // (traffic, wind, crowd, TV, construction) and surgically removes non-speech audio.
                const rnnoiseWasmBinary = await loadRnnoise({ url: rnnoiseWasmPath });
                await audioContext.audioWorklet.addModule(rnnoiseWorkletPath);
                const rnnoiseNode = new RnnoiseWorkletNode(audioContext, { wasmBinary: rnnoiseWasmBinary });
                console.log("🧠 AI Neural Noise Suppression (RNNoise) Active.");

                // Step 2.6: VOICE BANDPASS FILTER (300Hz - 3400Hz)
                // Extra safety layer on top of AI denoiser
                const highPassFilter = audioContext.createBiquadFilter();
                highPassFilter.type = 'highpass';
                highPassFilter.frequency.value = 300;
                highPassFilter.Q.value = 0.7;

                const lowPassFilter = audioContext.createBiquadFilter();
                lowPassFilter.type = 'lowpass';
                lowPassFilter.frequency.value = 3400;
                lowPassFilter.Q.value = 0.7;

                // Step 2.7: PCM Downsampler AudioWorklet (48kHz → 16kHz for Voice Engine)
                await audioContext.audioWorklet.addModule('/stt-worklet.js');
                const workletNode = new AudioWorkletNode(audioContext, 'stt-processor');

                // Chain: Mic → RNNoise AI → High-Pass → Low-Pass → PCM Downsampler
                const source = audioContext.createMediaStreamSource(micStream);
                source.connect(rnnoiseNode);           // Mic → AI Denoiser
                rnnoiseNode.connect(highPassFilter);    // AI Denoiser → Wind Filter
                highPassFilter.connect(lowPassFilter);  // Wind Filter → Hiss Filter
                lowPassFilter.connect(workletNode);     // Hiss Filter → 16kHz Downsampler
                workletNode.connect(audioContext.destination);
                console.log("🔇 Full Voice Isolation Pipeline Active: AI Denoise → Bandpass → 16kHz.");

                // Step 3: Connect to backend WebSocket proxy -> CX Voice Engine
                const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                const wsHost = window.location.host;
                const langCode = agent.language || 'hi-IN';
                sttWs = new WebSocket(`${wsProtocol}://${wsHost}/?type=stt&lang=${langCode}`);

                sttWs.onopen = () => {
                    console.log("🔗 Voice Engine WebSocket connected.");
                    setCallStatus('listening');
                };

                // Step 4: Pipe AudioWorklet PCM chunks as base64 to backend
                workletNode.port.onmessage = (event) => {
                    if (event.data.type === 'audio' && sttWs && sttWs.readyState === WebSocket.OPEN) {
                        const int16Array = new Int16Array(event.data.samples);
                        // Convert Int16Array to base64
                        const uint8 = new Uint8Array(int16Array.buffer);
                        let binaryStr = '';
                        for (let i = 0; i < uint8.length; i++) {
                            binaryStr += String.fromCharCode(uint8[i]);
                        }
                        const base64Chunk = window.btoa(binaryStr);
                        sttWs.send(JSON.stringify({ type: 'audio', chunk: base64Chunk }));
                    }
                };

                // Step 5: Handle transcription results from Voice Engine via backend proxy
                sttWs.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'transcript' && msg.text) {
                            const transcriptText = msg.text.trim();
                            if (!transcriptText) return;

                            if (msg.isFinal) {
                                finalTranscript += transcriptText + ' ';
                                interimTranscript = '';
                            } else {
                                interimTranscript = transcriptText;
                            }

                            const currentText = (finalTranscript + interimTranscript).trim();
                            const wordCount = currentText.split(/\s+/).filter(w => w.length > 0).length;
                            const isFillerPhrase = /^(yes|no|hello|hi|hey|yeah|yep|yup|okay|ok|uh huh|got it|sure|alright|right|correct|thanks|thank you|ha+|haan|han|ji|achha|acha|theek|sahi|na+|nahi|nahin|stop|wait|hold|pause|hmm+)[.,!?]?$/i.test(currentText);
                            const isStopText = /^(the|and|a|an|so|like|but|or|because|as|if|when|than|then|just|with|that|this|it|is|was|are|were)[.,!?]?$/i.test(currentText);

                            // Smart Barge-In
                            if (speakerActiveRef.current) {
                                if (wordCount >= 2 || (wordCount === 1 && !isStopText)) {
                                    if (audioRef.current) {
                                        if (aiCurrentSentenceRef.current) {
                                            historyRef.current = [...historyRef.current, { role: 'model', text: aiCurrentSentenceRef.current.trim() }];
                                            setHistory([...historyRef.current]);
                                        }
                                        aiCurrentSentenceRef.current = "";
                                        audioRef.current.pause();
                                        audioRef.current = null;
                                    }
                                    speakerActiveRef.current = false;
                                    setCallStatus('listening');
                                }
                            }

                            // Dynamic VAD Patience
                            let dynamicPatience = agent.patienceMs || 800;
                            if (wordCount > 0 && wordCount <= 3) dynamicPatience = 400;
                            if (isFillerPhrase) dynamicPatience = 200;

                            // Custom Silence Detection (VAD)
                            if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);

                            if (currentText.length >= 1) {
                                // Speculative Pre-fetching
                                if (predictiveTimer) clearTimeout(predictiveTimer);
                                if (wordCount >= 3 && currentText !== lastPredictedText) {
                                    predictiveTimer = setTimeout(async () => {
                                        try {
                                            lastPredictedText = currentText;
                                            activePredictionId = `pred_${Date.now()}_${Math.floor(Math.random()*1000)}`;
                                            console.log("🧠 Speculative AI pre-fetch triggered for:", currentText);
                                            await api.agentChatPredictive({
                                                agentId: agent.id,
                                                history: historyRef.current,
                                                userText: currentText,
                                                agentOverride: agent,
                                                predictionId: activePredictionId
                                            });
                                        } catch (err) {
                                            console.log("Prediction failed", err);
                                        }
                                    }, 150);
                                }

                                // Standard Silence Detection (VAD)
                                silenceTimerRef.current = setTimeout(async () => {
                                    const text = currentText;
                                    if (!text) return;
                                    finalTranscript = '';
                                    interimTranscript = '';

                                    const consumeId = (text === lastPredictedText) ? activePredictionId : null;
                                    if (consumeId) console.log("🚀 CONSUMING SPECULATIVE CACHE!");
                                    activePredictionId = null;
                                    lastPredictedText = '';

                                    await processSpeech(text, consumeId);
                                }, dynamicPatience);
                            }
                        } else if (msg.type === 'error') {
                            console.error("Voice Engine Error:", msg.message);
                        }
                    } catch (e) {
                        console.error("STT WS message parse error:", e);
                    }
                };

                sttWs.onerror = (err) => {
                    console.error("STT WebSocket Error:", err);
                };

                sttWs.onclose = () => {
                    console.log("STT WebSocket closed.");
                };

            } catch (err) {
                console.error("Failed to initialize Voice Engine:", err);
                setCallStatus('error');
            }
        };

        initCxSTT();

        // Simulated ring delay before outbound greeting starts
        setTimeout(() => {
            processSpeech(''); // Empty string triggers outbound greeting
        }, 1500);

        return () => {
            if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
            if (predictiveTimer) clearTimeout(predictiveTimer);
            isModalOpenRef.current = false;
            if (audioRef.current) {
                audioRef.current.pause();
            }
            if (sttWs && sttWs.readyState === WebSocket.OPEN) {
                sttWs.close();
            }
            if (audioContext && audioContext.state !== 'closed') {
                audioContext.close();
            }
            if (micStream) {
                micStream.getTracks().forEach(t => t.stop());
            }
        };
    }, []);

    const processSpeech = async (userText, predictionId = null) => {
        const reqId = ++activeReqIdRef.current;

        // Barge-in: if AI was speaking, interrupt it
        if (speakerActiveRef.current && audioRef.current) {
            audioRef.current.pause();
            audioRef.current = null;
        }

        speakerActiveRef.current = true;
        setCallStatus('ai-speaking');

        let currentHistory = [...historyRef.current];
        if (userText) {
            currentHistory.push({ role: 'user', text: userText });
            historyRef.current = currentHistory;
            setHistory([...currentHistory]); // Track context silently
        }

        try {
            const res = await api.agentChatStream({
                agentId: agent.id,
                history: currentHistory,
                userText: userText,
                agentOverride: agent,
                predictionId: predictionId
            });

            if (activeReqIdRef.current !== reqId) return;

            if (!res.ok) throw new Error("API Failed");

            // Play streaming audio via MediaSource (MSE) for ultra-low latency multiplexed over NDJSON
            const mediaSource = new MediaSource();
            const url = URL.createObjectURL(mediaSource);
            const audio = new Audio(url);
            audio.playbackRate = agent.prosodyRate ?? 1.0; // Apply custom voice speed
            audioRef.current = audio;

            triggerListen();

            await new Promise((resolve, reject) => {
                mediaSource.addEventListener('sourceopen', async () => {
                    try {
                        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
                        const reader = res.body.getReader();
                        let textDecoder = new TextDecoder();
                        let bufferString = "";
                        
                        let appendQueue = [];
                        let isAppending = false;

                        const pumpQueue = () => {
                            if (isAppending || appendQueue.length === 0 || mediaSource.readyState !== 'open') return;
                            isAppending = true;
                            const bytes = appendQueue.shift();
                            
                            const onUpdateEnd = () => {
                                sourceBuffer.removeEventListener('updateend', onUpdateEnd);
                                isAppending = false;
                                pumpQueue();
                            };
                            
                            sourceBuffer.addEventListener('updateend', onUpdateEnd);
                            try {
                                sourceBuffer.appendBuffer(bytes);
                            } catch (e) {
                                console.error("MSE append error", e);
                                isAppending = false;
                            }
                        };
                        
                        const pump = async () => {
                            if (activeReqIdRef.current !== reqId) return;
                            const { done, value } = await reader.read();
                            
                            if (done) {
                                if (bufferString.trim()) {
                                    try {
                                        const event = JSON.parse(bufferString.trim());
                                        if (event.type === 'text') aiCurrentSentenceRef.current += event.data;
                                    } catch (e) {}
                                }
                                
                                const endInterval = setInterval(() => {
                                    if (appendQueue.length === 0 && !isAppending && !sourceBuffer.updating) {
                                        clearInterval(endInterval);
                                        if (mediaSource.readyState === 'open') mediaSource.endOfStream();
                                    }
                                }, 50);
                                return;
                            }
                            
                            bufferString += textDecoder.decode(value, { stream: true });
                            const lines = bufferString.split('\n');
                            bufferString = lines.pop(); // Keep incomplete line

                            for (const line of lines) {
                                if (!line.trim()) continue;
                                try {
                                    const event = JSON.parse(line);
                                    if (event.type === 'text') {
                                        aiCurrentSentenceRef.current += event.data;
                                    } else if (event.type === 'audio') {
                                        const binaryStr = window.atob(event.data);
                                        const len = binaryStr.length;
                                        const bytes = new Uint8Array(len);
                                        for (let i = 0; i < len; i++) bytes[i] = binaryStr.charCodeAt(i);
                                        
                                        appendQueue.push(bytes);
                                        pumpQueue();
                                    }
                                } catch (err) { }
                            }
                            
                            pump();
                        };
                        
                        pump();
                    } catch (err) {
                        console.error("MSE format unsupported?", err);
                        reject(err);
                    }
                });

                audio.onended = () => {
                    URL.revokeObjectURL(url);
                    if (aiCurrentSentenceRef.current) {
                        historyRef.current = [...historyRef.current, { role: 'model', text: aiCurrentSentenceRef.current.trim() }];
                        setHistory([...historyRef.current]);
                    }
                    aiCurrentSentenceRef.current = ""; // Reset
                    resolve();
                };
                audio.onerror = reject;
                audio.play().catch(reject);
            });

        } catch (e) {
            if (activeReqIdRef.current !== reqId) return;
            console.error(e);
            setCallStatus('error');
        } finally {
            if (activeReqIdRef.current === reqId) {
                speakerActiveRef.current = false;
                if (isModalOpenRef.current) {
                    setCallStatus('listening');
                    triggerListen();
                }
            }
        }
    };

    const handleEndCall = () => {
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        isModalOpenRef.current = false;
        recognitionRef.current?.abort();
        if (audioRef.current) audioRef.current.pause();
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-md z-[100] flex items-center justify-center animate-fade-in p-4">
            <div className="bg-white rounded-[2.5rem] shadow-2xl w-full max-w-sm overflow-hidden flex flex-col items-center py-12 px-6 relative border-4 border-gray-100">

                {/* Status Indicator */}
                <div className="absolute top-6 left-0 right-0 flex justify-center text-gray-400 text-xs tracking-widest uppercase font-semibold">
                    {callStatus === 'connecting' ? 'Connecting...' : callStatus === 'error' ? 'Call Failed' : 'Active Call'}
                </div>

                {/* Avatar */}
                <div className="relative mt-8 mb-6">
                    {callStatus === 'ai-speaking' && (
                        <div className="absolute inset-x-0 -inset-y-0.5 bg-orange-400/30 rounded-full blur-xl animate-pulse"></div>
                    )}
                    <div className={`w-32 h-32 rounded-full flex items-center justify-center relative z-10 shadow-lg transition-colors duration-500 ${callStatus === 'ai-speaking' ? 'bg-orange-500 text-white' : 'bg-gray-100 text-gray-400'}`}>
                        <Bot size={64} />
                    </div>
                </div>

                {/* Caller Info */}
                <div className="text-center space-y-2 mb-12">
                    <h2 className="text-3xl font-bold text-gray-900 max-w-[280px] truncate">{agent.name}</h2>
                    <div className="text-lg font-mono font-medium text-gray-500">
                        {callStatus === 'connecting' ? '...' : formatTime(duration)}
                    </div>
                </div>

                {/* Action State text */}
                <div className="h-8 mb-8 flex items-center justify-center">
                    {callStatus === 'listening' ? (
                        <div className="flex items-center gap-2 text-green-600 bg-green-50 px-4 py-1.5 rounded-full text-sm font-semibold animate-pulse">
                            <Mic size={16} /> Listening...
                        </div>
                    ) : callStatus === 'ai-speaking' ? (
                        <div className="flex items-center gap-2 text-orange-600 bg-orange-50 px-4 py-1.5 rounded-full text-sm font-semibold">
                            <Volume2 size={16} className="animate-pulse" /> Agent is speaking...
                        </div>
                    ) : null}
                </div>

                {/* End Call Button */}
                <button
                    onClick={handleEndCall}
                    className="w-20 h-20 rounded-full bg-red-500 hover:bg-red-600 shadow-xl shadow-red-500/30 flex items-center justify-center text-white transition-all hover:scale-105 active:scale-95"
                >
                    <PhoneCall size={32} className="rotate-[135deg]" />
                </button>
            </div>
        </div>
    );
}
