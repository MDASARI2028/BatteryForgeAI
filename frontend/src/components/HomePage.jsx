import React, { useState, useEffect, useRef } from 'react';
import {
    Activity, Terminal, Microscope, Monitor, Zap, Shield, ChevronRight,
    BarChart3, Database, Cpu, Radio, Layers, ArrowUpRight, Sparkles,
    Eye, Wrench, Factory, TrendingUp, CircuitBoard, Battery, Gauge
} from 'lucide-react';

// ============================================================================
// ANIMATED COUNTER HOOK
// ============================================================================
const useAnimatedCounter = (target, duration = 1200) => {
    const [count, setCount] = useState(0);
    const ref = useRef(null);

    useEffect(() => {
        const num = parseInt(target);
        if (isNaN(num)) return;
        let start = 0;
        const step = Math.max(1, Math.floor(num / (duration / 16)));
        ref.current = setInterval(() => {
            start += step;
            if (start >= num) {
                setCount(num);
                clearInterval(ref.current);
            } else {
                setCount(start);
            }
        }, 16);
        return () => clearInterval(ref.current);
    }, [target, duration]);

    return count;
};

// ============================================================================
// ANIMATED BACKGROUND GRID
// ============================================================================
const AnimatedGrid = () => (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Radial gradient glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-indigo-500/[0.07] rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[400px] bg-cyan-500/[0.05] rounded-full blur-[100px]" />
        <div className="absolute top-0 left-0 w-[400px] h-[300px] bg-purple-500/[0.04] rounded-full blur-[100px]" />

        {/* Grid lines */}
        <svg className="absolute inset-0 w-full h-full opacity-[0.04]">
            <defs>
                <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                    <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        {/* Floating particles */}
        {[...Array(6)].map((_, i) => (
            <div
                key={i}
                className="absolute w-1 h-1 bg-indigo-400/40 rounded-full"
                style={{
                    left: `${15 + i * 15}%`,
                    top: `${20 + (i % 3) * 25}%`,
                    animation: `float ${3 + i * 0.5}s ease-in-out infinite alternate`,
                    animationDelay: `${i * 0.4}s`
                }}
            />
        ))}
    </div>
);

// ============================================================================
// MINI ACTIVITY PULSE CHART
// ============================================================================
const ActivityPulse = () => {
    const [bars, setBars] = useState(Array(20).fill(0).map(() => Math.random() * 60 + 10));

    useEffect(() => {
        const interval = setInterval(() => {
            setBars(prev => {
                const next = [...prev];
                next.shift();
                next.push(Math.random() * 60 + 10);
                return next;
            });
        }, 400);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex items-end gap-[2px] h-8">
            {bars.map((h, i) => (
                <div
                    key={i}
                    className="w-[3px] bg-gradient-to-t from-indigo-600 to-cyan-400 rounded-full transition-all duration-300"
                    style={{ height: `${h}%`, opacity: 0.3 + (i / bars.length) * 0.7 }}
                />
            ))}
        </div>
    );
};

// ============================================================================
// STAT CARD WITH RADIAL GAUGE
// ============================================================================
const HeroStatCard = ({ icon: Icon, label, value, subtext, color, delay, isPercentage }) => {
    const numVal = parseInt(value) || 0;
    const displayValue = useAnimatedCounter(numVal);

    return (
        <div
            className="group relative animate-in fade-in slide-in-from-bottom-6 fill-mode-backwards"
            style={{ animationDelay: `${delay}ms` }}
        >
            {/* Gradient border effect */}
            <div className={`absolute -inset-[1px] bg-gradient-to-br from-${color}-500/30 via-transparent to-${color}-500/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm`} />

            <div className={`relative bg-slate-900/60 backdrop-blur-xl rounded-2xl p-6 border border-white/[0.06] overflow-hidden transition-all duration-500 group-hover:border-${color}-500/30 group-hover:shadow-lg group-hover:shadow-${color}-500/10`}>
                {/* Background glow */}
                <div className={`absolute -right-6 -top-6 w-28 h-28 bg-${color}-500/10 rounded-full blur-2xl group-hover:bg-${color}-500/20 transition-all duration-700`} />

                <div className="relative z-10">
                    <div className="flex items-center justify-between mb-4">
                        <div className={`p-2.5 bg-${color}-500/10 rounded-xl border border-${color}-500/20 group-hover:scale-110 transition-transform duration-300`}>
                            <Icon className={`w-5 h-5 text-${color}-400`} />
                        </div>
                        <ActivityPulse />
                    </div>

                    <div className="flex items-end gap-2 mb-1">
                        <span className="text-4xl font-bold text-white font-mono tracking-tight">
                            {typeof value === 'string' && value.includes('%') ? displayValue + '%' : displayValue}
                        </span>
                        {subtext && (
                            <span className={`text-xs px-2 py-0.5 rounded-full bg-${color}-500/10 text-${color}-300 border border-${color}-500/20 font-medium mb-1`}>
                                {subtext}
                            </span>
                        )}
                    </div>
                    <p className="text-sm text-slate-400 font-medium">{label}</p>
                </div>
            </div>
        </div>
    );
};

// ============================================================================
// FEATURE CARD (REPLACES QUICK ACTIONS)
// ============================================================================
const FeatureCard = ({ icon: Icon, title, desc, onClick, color, delay, tag }) => (
    <button
        onClick={onClick}
        className={`group relative w-full text-left animate-in fade-in slide-in-from-bottom-4 fill-mode-backwards`}
        style={{ animationDelay: `${delay}ms` }}
    >
        {/* Hover glow */}
        <div className={`absolute -inset-[1px] bg-gradient-to-r from-${color}-500/20 to-${color}-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-all duration-500`} />

        <div className={`relative bg-slate-900/50 backdrop-blur-sm border border-white/[0.06] rounded-2xl p-5 transition-all duration-300 group-hover:border-${color}-500/30 group-hover:bg-slate-800/60 overflow-hidden`}>
            {/* Accent line */}
            <div className={`absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-${color}-500/50 to-transparent scale-x-0 group-hover:scale-x-100 transition-transform duration-500`} />

            <div className="flex items-start gap-4">
                <div className={`shrink-0 p-3 bg-${color}-500/10 rounded-xl border border-${color}-500/20 group-hover:scale-110 group-hover:bg-${color}-500/20 transition-all duration-300`}>
                    <Icon className={`w-6 h-6 text-${color}-400`} />
                </div>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <h4 className="text-base font-semibold text-white group-hover:text-white transition-colors">{title}</h4>
                        {tag && (
                            <span className={`text-[9px] px-1.5 py-0.5 rounded bg-${color}-500/10 text-${color}-300 border border-${color}-500/20 uppercase font-bold tracking-wider`}>
                                {tag}
                            </span>
                        )}
                    </div>
                    <p className="text-sm text-slate-500 group-hover:text-slate-400 transition-colors leading-relaxed">{desc}</p>
                </div>
                <ArrowUpRight className={`w-5 h-5 text-slate-600 group-hover:text-${color}-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all duration-300 shrink-0 mt-1`} />
            </div>
        </div>
    </button>
);

// ============================================================================
// MAIN HOMEPAGE COMPONENT
// ============================================================================
const HomePage = ({ onNavigate }) => {
    const [stats, setStats] = useState({ active: "0", health: "0", events: "0", sim_hours: "24" });
    const [loading, setLoading] = useState(true);
    const [logs, setLogs] = useState([]);
    const [wsStatus, setWsStatus] = useState("Connecting...");

    // Fetch fleet stats
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/fleet/data`);
                if (res.ok) {
                    const data = await res.json();
                    if (data.data && data.data.fleet_metrics) {
                        setStats({
                            active: String(data.data.fleet_metrics.active_packs || 0),
                            health: String(data.data.fleet_metrics.avg_health || 0),
                            events: String(data.data.red_list ? data.data.red_list.filter(r => r.status === 'CRITICAL').length : 0),
                            sim_hours: "24"
                        });
                    }
                }
            } catch (e) {
                console.error("Home fetch error", e);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    // WebSocket live logs
    useEffect(() => {
        const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/api/ws/logs`);
        ws.onopen = () => {
            setWsStatus("Connected");
            setLogs(prev => [`[SYS] Connected to Live Telemetry`, ...prev]);
        };
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const formattedLog = `[${data.timestamp}] ${data.system}: ${data.message}`;
                setLogs(prev => [formattedLog, ...prev].slice(0, 8));
            } catch (e) { /* ignore parse errors */ }
        };
        ws.onclose = () => {
            setWsStatus("Disconnected");
            setLogs(prev => [`[SYS] Stream Connection Lost`, ...prev]);
        };
        return () => ws.close();
    }, []);

    return (
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-10">
            {/* Animated background */}
            <AnimatedGrid />

            {/* CSS Animations */}
            <style>{`
                @keyframes float {
                    0% { transform: translateY(0px) scale(1); opacity: 0.4; }
                    100% { transform: translateY(-20px) scale(1.5); opacity: 0.1; }
                }
                @keyframes gradient-x {
                    0%, 100% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                }
                @keyframes pulse-ring {
                    0% { transform: scale(0.8); opacity: 1; }
                    100% { transform: scale(2); opacity: 0; }
                }
            `}</style>

            {/* ============ HERO SECTION ============ */}
            <div className="relative mb-10 md:mb-16 animate-in fade-in zoom-in-95 duration-700">
                <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-8">
                    <div className="max-w-3xl">
                        {/* Brand */}
                        <div className="flex items-center gap-4 mb-6">
                            <div className="relative shrink-0">
                                <div className="p-3 bg-gradient-to-br from-indigo-600 to-cyan-600 rounded-2xl shadow-lg shadow-indigo-500/25">
                                    <CircuitBoard className="w-8 h-8 text-white" />
                                </div>
                                {/* Animated ring */}
                                <div className="absolute inset-0 rounded-2xl border-2 border-indigo-400/50" style={{ animation: 'pulse-ring 2s ease-out infinite' }} />
                            </div>
                            <div>
                                <h1 className="text-3xl md:text-5xl font-bold text-white tracking-tight">
                                    BatteryForge{' '}
                                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-cyan-400 to-indigo-400 bg-[length:200%_auto]" style={{ animation: 'gradient-x 4s ease infinite' }}>
                                        AI
                                    </span>
                                </h1>
                                <p className="text-xs md:text-sm text-slate-500 font-medium uppercase tracking-widest">
                                    Autonomous Intelligence Platform
                                </p>
                            </div>
                        </div>

                        <p className="text-base md:text-xl text-slate-400 leading-relaxed mb-8 max-w-2xl">
                            End-to-end battery and PCB manufacturing intelligence powered by{' '}
                            <span className="text-indigo-400 font-medium">Gemini 3 Pro</span> — from defect detection to fleet-scale predictive analytics.
                        </p>

                        {/* Quick launch buttons */}
                        <div className="flex flex-wrap gap-3">
                            <button
                                onClick={() => onNavigate('pcb')}
                                className="group px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 rounded-xl text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 transition-all flex items-center gap-2"
                            >
                                <Layers className="w-4 h-4" />
                                BMS & Pack Assembly
                                <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                            </button>
                            <button
                                onClick={() => onNavigate('fleet')}
                                className="px-6 py-3 bg-slate-800/80 hover:bg-slate-700/80 border border-white/10 rounded-xl text-slate-300 font-semibold text-sm transition-all flex items-center gap-2"
                            >
                                <Monitor className="w-4 h-4" />
                                Fleet Monitor
                            </button>
                        </div>
                    </div>

                    {/* System status beacon - Hidden on small screens, shown on large */}
                    <div className="hidden lg:block shrink-0">
                        <div className="bg-slate-900/60 backdrop-blur-xl border border-white/[0.06] rounded-2xl p-6 w-72">
                            <div className="flex items-center gap-3 mb-5">
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                                </span>
                                <span className="text-sm font-bold text-emerald-400 tracking-wide uppercase">All Systems Online</span>
                            </div>
                            <div className="space-y-4">
                                {[
                                    { label: 'Gemini 3 Pro', status: 'Active', ok: true },
                                    { label: 'Physics Engine', status: 'Running', ok: true },
                                    { label: 'Vision Pipeline', status: 'Ready', ok: true },
                                    { label: 'WebSocket', status: wsStatus, ok: wsStatus === 'Connected' },
                                ].map((s, i) => (
                                    <div key={i} className="flex items-center justify-between border-b border-white/5 pb-2 last:border-0 last:pb-0">
                                        <span className="text-xs text-slate-500 font-medium">{s.label}</span>
                                        <span className={`text-xs font-bold ${s.ok ? 'text-emerald-400' : 'text-amber-400'}`}>
                                            {s.status}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ============ STATS GRID ============ */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-12">
                <HeroStatCard
                    icon={Radio}
                    label="Active Fleet Packs"
                    value={loading ? "0" : stats.active}
                    subtext="Live"
                    color="emerald"
                    delay={100}
                />
                <HeroStatCard
                    icon={Activity}
                    label="Avg Fleet Health"
                    value={loading ? "0" : stats.health}
                    subtext={parseInt(stats.health) < 90 ? "Degraded" : "Stable"}
                    color={parseInt(stats.health) < 90 ? "amber" : "indigo"}
                    delay={200}
                    isPercentage
                />
                <HeroStatCard
                    icon={Shield}
                    label="Safety Events"
                    value={loading ? "0" : stats.events}
                    subtext={parseInt(stats.events) > 0 ? "Alert" : "Clear"}
                    color={parseInt(stats.events) > 0 ? "red" : "cyan"}
                    delay={300}
                />
                <HeroStatCard
                    icon={Cpu}
                    label="Simulation Window"
                    value={stats.sim_hours}
                    subtext="PyBaMM"
                    color="purple"
                    delay={400}
                />
            </div>

            {/* ============ MAIN CONTENT ============ */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Feature Cards */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="flex items-center gap-4 mb-2 animate-in fade-in duration-500">
                        <h2 className="text-xl font-bold text-white tracking-tight">Intelligence Workspaces</h2>
                        <div className="flex-1 h-px bg-gradient-to-r from-white/10 via-white/5 to-transparent" />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-5">
                        <FeatureCard
                            icon={Microscope}
                            title="Visual Intelligence"
                            desc="AI defect detection, thermal scout, and live video analysis"
                            color="indigo"
                            delay={400}
                            tag="Vision"
                            onClick={() => onNavigate('visual')}
                        />
                        <FeatureCard
                            icon={Layers}
                            title="BMS & Pack Assembly"
                            desc="Formation optimization, welding control, vision QC, and line monitoring"
                            color="purple"
                            delay={500}
                            tag="Gemini 3"
                            onClick={() => onNavigate('pcb')}
                        />
                        <FeatureCard
                            icon={Zap}
                            title="Simulation Lab"
                            desc="DFN physics models, aging prediction, and charging analysis"
                            color="emerald"
                            delay={600}
                            tag="PyBaMM"
                            onClick={() => onNavigate('sim')}
                        />
                        <FeatureCard
                            icon={Database}
                            title="Fleet Monitor"
                            desc="Real-time EV fleet telemetry, thermal mapping, and alerts"
                            color="rose"
                            delay={700}
                            tag="Live"
                            onClick={() => onNavigate('fleet')}
                        />
                        <FeatureCard
                            icon={Terminal}
                            title="Log Analysis"
                            desc="Semantic BMS error log parsing with AI-powered diagnostics"
                            color="amber"
                            delay={800}
                            onClick={() => onNavigate('logs')}
                        />
                    </div>
                </div>

                {/* Right Sidebar - Live Feed + Capabilities */}
                <div className="space-y-6">
                    {/* Live System Stream */}
                    <div className="bg-slate-900/50 backdrop-blur-xl border border-white/[0.06] rounded-2xl p-6 animate-in fade-in slide-in-from-right duration-700 delay-500">
                        <div className="flex items-center justify-between mb-5">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2 uppercase tracking-wider">
                                <Radio className="w-4 h-4 text-emerald-500 animate-pulse" />
                                Live Telemetry
                            </h3>
                            <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold uppercase ${wsStatus === 'Connected' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                }`}>{wsStatus}</span>
                        </div>

                        <div className="bg-black/50 rounded-xl p-4 font-mono text-[11px] h-52 overflow-hidden border border-white/5 relative">
                            <div className="space-y-2">
                                {logs.length === 0 ? (
                                    <div className="text-slate-600 flex flex-col items-center gap-3 py-12 justify-center">
                                        <Radio className="w-6 h-6 animate-pulse" />
                                        <span className="text-xs">Awaiting telemetry stream...</span>
                                    </div>
                                ) : (
                                    logs.map((log, i) => (
                                        <div
                                            key={i}
                                            className="text-emerald-500/80 animate-in fade-in slide-in-from-left-2 duration-200 flex items-start gap-2"
                                            style={{ opacity: 1 - i * 0.12 }}
                                        >
                                            <span className="text-indigo-500/60 select-none shrink-0 font-bold">{'>'}</span>
                                            <span className="break-all leading-relaxed">{log}</span>
                                        </div>
                                    ))
                                )}
                            </div>
                            <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-black/80 to-transparent pointer-events-none" />
                        </div>
                    </div>

                    {/* AI Capabilities Card */}
                    <div className="bg-gradient-to-br from-indigo-950/40 to-slate-900/60 backdrop-blur-xl border border-indigo-500/10 rounded-2xl p-6 animate-in fade-in slide-in-from-right duration-700 delay-700">
                        <div className="flex items-center gap-2 mb-5">
                            <Sparkles className="w-4 h-4 text-indigo-400" />
                            <h3 className="text-sm font-bold text-white uppercase tracking-wider">AI Core Capabilities</h3>
                        </div>
                        <div className="grid grid-cols-1 gap-2">
                            {[
                                { icon: Eye, label: 'Vision Defect Detection', color: 'purple' },
                                { icon: Wrench, label: 'Predictive Maintenance', color: 'orange' },
                                { icon: TrendingUp, label: 'Battery Aging Forecast', color: 'emerald' },
                                { icon: Factory, label: 'Process Control Loop', color: 'cyan' },
                                { icon: Shield, label: 'Safety Event Analysis', color: 'red' },
                            ].map((cap, i) => (
                                <div key={i} className="flex items-center gap-3 p-2.5 rounded-xl hover:bg-white/5 transition-all group border border-transparent hover:border-white/5">
                                    <div className={`p-2 bg-${cap.color}-500/10 rounded-lg group-hover:scale-110 transition-transform`}>
                                        <cap.icon className={`w-4 h-4 text-${cap.color}-400`} />
                                    </div>
                                    <span className="text-xs font-medium text-slate-400 group-hover:text-slate-200 transition-colors">{cap.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
