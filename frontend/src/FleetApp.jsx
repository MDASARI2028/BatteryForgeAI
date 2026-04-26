import React, { useState, useEffect } from 'react';
import { FleetProvider } from './components/fleet/shared/FleetContext';
import FleetDashboard from './components/fleet/FleetDashboard';
import VehicleManagement from './components/fleet/VehicleManagement';
import DriverManagement from './components/fleet/DriverManagement';
import FleetAnalytics from './components/fleet/FleetAnalytics';
import ChargingManagement from './components/fleet/ChargingManagement';
import RouteManagement from './components/fleet/RouteManagement';
import FleetSettings from './components/fleet/FleetSettings';
import FleetMonitor from './components/FleetMonitor';
import {
    Home, Car, Users, BarChart3, Zap, Route, Wrench,
    Settings, Menu, X, Activity
} from 'lucide-react';

/**
 * Main EV Fleet Management Application
 * Phase 1: Added fleet data callbacks for agent context integration
 */
function FleetApp({ onFleetDataUpdate, onProactiveAlert }) {
    const [activeView, setActiveView] = useState('dashboard');
    const [sidebarOpen, setSidebarOpen] = useState(true);

    const navigation = [
        { id: 'dashboard', name: 'Mission Control', icon: Home },
        { id: 'telemetry', name: 'Telemetry', icon: Activity },
        { id: 'vehicles', name: 'Vehicles', icon: Car },
        { id: 'drivers', name: 'Drivers', icon: Users },
        { id: 'analytics', name: 'Analytics', icon: BarChart3 },
        { id: 'charging', name: 'Charging', icon: Zap },
        { id: 'routes', name: 'Routes', icon: Route },
        { id: 'maintenance', name: 'Maintenance', icon: Wrench },
        { id: 'settings', name: 'Settings', icon: Settings },
    ];

    const renderView = () => {
        switch (activeView) {
            case 'dashboard':
                return <FleetDashboard />;
            case 'telemetry':
                return <FleetMonitor onFleetDataUpdate={onFleetDataUpdate} onProactiveAlert={onProactiveAlert} />;
            case 'vehicles':
                return <VehicleManagement />;
            case 'drivers':
                return <DriverManagement />;
            case 'analytics':
                return <FleetAnalytics />;
            case 'charging':
                return <ChargingManagement />;
            case 'routes':
                return <RouteManagement />;
            case 'settings':
                return <FleetSettings />;
            default:
                return (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <h2 className="text-2xl font-bold text-white mb-2">{activeView} Module</h2>
                            <p className="text-slate-400">Coming soon...</p>
                        </div>
                    </div>
                );
        }
    };

    return (
        <FleetProvider>
            <div className="min-h-[calc(100vh-3.5rem)] bg-gradient-to-br from-slate-950 via-slate-900 to-black flex relative">
                {/* Mobile Sidebar Overlay */}
                {sidebarOpen && (
                    <div
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 lg:hidden"
                        onClick={() => setSidebarOpen(false)}
                    />
                )}

                {/* Sidebar */}
                <aside className={`fixed inset-y-0 left-0 z-40 lg:relative lg:translate-x-0 transition-transform duration-300 ${sidebarOpen ? 'translate-x-0 w-64' : '-translate-x-full lg:translate-x-0 w-20'} bg-slate-900/90 lg:bg-slate-900/50 border-r border-white/5 flex flex-col`}>
                    {/* Logo */}
                    <div className="p-4 border-b border-white/5 flex items-center justify-between">
                        {(sidebarOpen || window.innerWidth < 1024) && (
                            <div>
                                <h1 className="text-xl font-bold text-white">BatteryForge</h1>
                                <p className="text-xs text-slate-400">EV Fleet Monitor</p>
                            </div>
                        )}
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                        >
                            {sidebarOpen ? <X className="w-5 h-5 text-slate-400" /> : <Menu className="w-5 h-5 text-slate-400" />}
                        </button>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            const isActive = activeView === item.id;

                            return (
                                <button
                                    key={item.id}
                                    onClick={() => {
                                        setActiveView(item.id);
                                        if (window.innerWidth < 1024) setSidebarOpen(false);
                                    }}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${isActive
                                        ? 'bg-blue-600 text-white'
                                        : 'text-slate-400 hover:bg-white/5 hover:text-white'
                                        }`}
                                    title={!sidebarOpen ? item.name : ''}
                                >
                                    <Icon className="w-5 h-5 shrink-0" />
                                    {(sidebarOpen || window.innerWidth < 1024) && <span className="font-medium">{item.name}</span>}
                                </button>
                            );
                        })}
                    </nav>

                    {/* Footer */}
                    <div className="p-4 border-t border-white/5">
                        <div className="flex items-center gap-2 text-xs text-slate-500">
                            <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                            {(sidebarOpen || window.innerWidth < 1024) && <span>System Online</span>}
                        </div>
                    </div>
                </aside>

                {/* Main Content */}
                <main className="flex-1 overflow-auto relative">
                    {/* Mobile Sidebar Toggle (Floating when sidebar closed) */}
                    {!sidebarOpen && (
                        <button
                            onClick={() => setSidebarOpen(true)}
                            className="lg:hidden fixed bottom-20 left-4 z-20 p-3 bg-slate-800 text-white rounded-full shadow-lg border border-white/10"
                        >
                            <Menu className="w-6 h-6" />
                        </button>
                    )}

                    <div className="p-4 md:p-6">
                        {renderView()}
                    </div>
                </main>
            </div>
        </FleetProvider>
    );
}

export default FleetApp;
