import React, { useState } from 'react';
import UploadZone from './components/UploadZone';
import LogParser from './components/LogParser';
import AnalysisResult from './components/AnalysisResult';
import ChatInterface from './components/ChatInterface';
import ChargingAnalysis from './components/ChargingAnalysis';
import DisclaimerModal from './components/DisclaimerModal';
import { WorkspaceProvider } from './context/WorkspaceContext';

// import FleetMonitor from './components/FleetMonitor';
import FleetApp from './FleetApp';
import VisualIntelligence from './components/VisualIntelligence';
import PCBManufacturing from './components/PCBManufacturing';
import HomePage from './components/HomePage';

import { Microscope, Activity, Cpu, Monitor, Terminal, Home, Layers } from 'lucide-react';

function App() {
    const [logResult, setLogResult] = useState(null);
    const [activeWorkspace, setActiveWorkspace] = useState('home'); // 'home', 'visual', 'logs', 'sim', 'fleet'
    const [agingMetrics, setAgingMetrics] = useState(null);
    const [alertLevel, setAlertLevel] = useState('normal'); // 'normal', 'critical'
    const [isAgentOpen, setIsAgentOpen] = useState(false); // UI State for Layout Occlusion Fix
    const [agentTraces, setAgentTraces] = useState([]); // Real-time agent trace data
    const [isAgentActive, setIsAgentActive] = useState(false); // Agent processing indicator
    const [disclaimerAccepted, setDisclaimerAccepted] = useState(false); // Disclaimer acceptance

    // Phase 1: Fleet Data Bridge - Live fleet data from FleetMonitor
    const [liveFleetData, setLiveFleetData] = useState(null);
    const [proactiveAlerts, setProactiveAlerts] = useState([]); // Phase 4: Proactive alerts queue

    // ADK Standard: Formal Agent State Object with Fleet Context
    const agentState = {
        session_id: "demo-session-001",
        workspace: {
            active_tab: activeWorkspace,
            is_loading: false,
            alert_level: alertLevel
        },
        context: {
            // Grounding Data
            log_analysis: logResult,
            telemetry: agingMetrics,
            // Phase 1: Live Fleet Data for Agent Awareness
            fleet: liveFleetData ? {
                health: liveFleetData.data?.fleet_metrics?.avg_health,
                thermal_spread: liveFleetData.data?.fleet_metrics?.thermal_spread,
                critical_count: liveFleetData.data?.red_list?.length,
                active_packs: liveFleetData.data?.fleet_metrics?.active_packs,
                red_list: liveFleetData.data?.red_list,
                commander_report: liveFleetData.commander_report,
                vehicles: liveFleetData.vehicles,
                drivers: liveFleetData.drivers
            } : null
        }
    };

    // This function allows the Agent (ChatInterface) to change the workspace view
    const handleAgentAction = (action) => {
        if (action === 'show_home') setActiveWorkspace('home');
        if (action === 'show_visual') setActiveWorkspace('visual');
        if (action === 'show_logs') setActiveWorkspace('logs');
        if (action === 'show_sim') setActiveWorkspace('sim');
        if (action === 'show_fleet') setActiveWorkspace('fleet');
        if (action === 'show_pcb') setActiveWorkspace('pcb');
        // 'show_visual_scout' now maps to the 'visual' workspace as it's merged
        if (action === 'show_visual_scout') setActiveWorkspace('visual');

        // Safety Actions
        if (action === 'trigger_red_alert') setAlertLevel('critical');
        if (action === 'clear_alert') setAlertLevel('normal');
    };

    // Phase 4: Handle proactive alerts from FleetMonitor
    const handleProactiveAlert = (alert) => {
        // Deduplicate alerts by pack_id within last 30 seconds
        setProactiveAlerts(prev => {
            const thirtySecondsAgo = Date.now() - 30000;
            const filtered = prev.filter(a =>
                new Date(a.timestamp).getTime() > thirtySecondsAgo
            );
            // Check if this pack already has an alert
            const exists = filtered.some(a => a.pack_id === alert.pack_id);
            if (exists) return filtered;
            return [...filtered, alert];
        });
    };

    return (
        <div className="h-screen bg-gray-50 text-gray-900 overflow-hidden flex flex-col font-sans relative">

            {/* DISCLAIMER MODAL - Shows on first load */}
            <DisclaimerModal onAccept={() => setDisclaimerAccepted(true)} />

            {/* RED ALERT OVERLAY */}
            {alertLevel === 'critical' && (
                <div className="fixed inset-0 z-50 pointer-events-none">
                    <div className="absolute inset-0 bg-red-500/10"></div>
                    <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-600 text-white font-bold text-lg px-6 py-3 rounded-lg shadow-lg border border-red-500">
                        ⚠️ CRITICAL ALERT ⚠️
                    </div>
                </div>
            )}

            {/* Clean Header */}
            <header className={`h-14 border-b border-gray-200 bg-white flex items-center justify-between px-6 shrink-0 z-20 ${alertLevel === 'critical' ? 'bg-red-50 border-red-200' : ''}`}>
                <div className="flex items-center gap-3">
                    <div className="p-1.5 bg-blue-100 rounded-md">
                        <Cpu className="text-blue-600 w-4 h-4" />
                    </div>
                    <span className="font-bold text-lg text-gray-900">
                        BatteryForge<span className="text-gray-500 font-normal">AI</span>
                    </span>
                    <span className="text-xs uppercase tracking-wide text-green-600 font-medium ml-2 px-2 py-0.5 bg-green-100 rounded">
                        Online
                    </span>
                </div>
                <div className="flex gap-1 bg-gray-100 p-1 rounded-md">
                    <button onClick={() => setActiveWorkspace('home')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'home' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="Home"><Home className="w-4 h-4" /></button>
                    <button onClick={() => setActiveWorkspace('visual')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'visual' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="Visual Intelligence"><Microscope className="w-4 h-4" /></button>
                    <button onClick={() => setActiveWorkspace('logs')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'logs' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="Logs"><Terminal className="w-4 h-4" /></button>
                    <button onClick={() => setActiveWorkspace('sim')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'sim' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="Simulations"><Activity className="w-4 h-4" /></button>
                    <button onClick={() => setActiveWorkspace('fleet')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'fleet' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="Fleet Monitor"><Monitor className="w-4 h-4" /></button>
                    <button onClick={() => setActiveWorkspace('pcb')} className={`p-1.5 rounded transition-colors ${activeWorkspace === 'pcb' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'}`} title="PCB Factory"><Layers className="w-4 h-4" /></button>
                </div>
            </header>

            {/* Split Screen Layout */}
            <div className="flex-1 flex overflow-hidden relative">


                {/* FLOATING AGENT (Result of UI Redesign) */}
                {/* We pass the state setter down so ChatInterface can control it, but App controls the layout */}
                <ChatInterface
                    onAction={handleAgentAction}
                    agentState={agentState}
                    externalOpenState={isAgentOpen}
                    setExternalOpenState={setIsAgentOpen}
                    onTraceUpdate={setAgentTraces}
                    onActivityChange={setIsAgentActive}
                    activeWorkspace={activeWorkspace}
                    liveFleetData={liveFleetData}
                    proactiveAlerts={proactiveAlerts}
                    onClearAlert={(alertId) => setProactiveAlerts(prev => prev.filter(a => a.id !== alertId))}
                />

                {/* MAIN WORKSPACE */}
                {/* Dynamically adjust margin when agent is open to prevent occlusion */}
                <main
                    className={`flex-1 bg-gray-50 p-6 overflow-y-auto relative transition-all duration-300 ease-in-out ${isAgentOpen ? 'mr-[460px]' : 'mr-0'}`}
                >
                    <div className="max-w-7xl mx-auto h-full flex flex-col relative">

                        {/* HOME DASHBOARD */}
                        {activeWorkspace === 'home' && (
                            <HomePage onNavigate={setActiveWorkspace} />
                        )}

                        {activeWorkspace === 'visual' && (
                            <VisualIntelligence />
                        )}

                        {activeWorkspace === 'logs' && (
                            <div className="animate-in fade-in duration-300">
                                <div className="flex items-center gap-4 mb-6">
                                    <div className="p-3 bg-blue-100 rounded-lg text-blue-600"><Terminal className="w-6 h-6" /></div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-gray-900">Log Analysis</h2>
                                        <p className="text-sm text-gray-600">Semantic Parsing of BMS Error Logs</p>
                                    </div>
                                </div>
                                <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                                    <LogParser onResult={setLogResult} context={agingMetrics} />
                                    <AnalysisResult data={logResult} type="log" />
                                </div>
                            </div>
                        )}

                        {activeWorkspace === 'sim' && (
                            <div className="animate-in fade-in duration-300 h-full">
                                <div className="flex items-center gap-4 mb-6">
                                    <div className="p-3 bg-blue-100 rounded-lg text-blue-600"><Cpu className="w-6 h-6" /></div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-gray-900">Simulation Laboratory</h2>
                                        <p className="text-sm text-gray-600">Generative AI & Aging Prediction Models</p>
                                    </div>
                                </div>
                                <div className="flex flex-col gap-6 pb-10">
                                    <ChargingAnalysis onAnalysisComplete={setAgingMetrics} />
                                </div>
                            </div>
                        )}

                        {activeWorkspace === 'fleet' && (
                            <div className="animate-in fade-in duration-300 h-full flex flex-col">
                                <div className="flex-1 h-full -m-6">
                                    <FleetApp
                                        onFleetDataUpdate={setLiveFleetData}
                                        onProactiveAlert={handleProactiveAlert}
                                    />
                                </div>
                            </div>
                        )}

                        {activeWorkspace === 'pcb' && (
                            <PCBManufacturing />
                        )}
                    </div>
                </main>
            </div >
        </div >
    )
}

export default App;
