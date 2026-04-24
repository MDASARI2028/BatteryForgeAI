import React from 'react';
import {
    Activity, Terminal, Microscope, Monitor, Cpu, Layers,
    BarChart3, Database, Radio, TrendingUp, Battery, Gauge
} from 'lucide-react';

// ============================================================================
// SIMPLE STAT CARD
// ============================================================================
const StatCard = ({ icon: Icon, label, value, color }) => (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
        <div className="flex items-center gap-4">
            <div className={`p-3 bg-${color}-100 rounded-lg`}>
                <Icon className={`w-6 h-6 text-${color}-600`} />
            </div>
            <div>
                <div className="text-2xl font-bold text-gray-900">{value}</div>
                <div className="text-sm text-gray-600">{label}</div>
            </div>
        </div>
    </div>
);

// ============================================================================
// FEATURE CARD
// ============================================================================
const FeatureCard = ({ icon: Icon, title, desc, onClick, color }) => (
    <button
        onClick={onClick}
        className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-all hover:border-gray-300 text-left group"
    >
        <div className="flex items-start gap-4">
            <div className={`p-3 bg-${color}-100 rounded-lg group-hover:bg-${color}-200 transition-colors`}>
                <Icon className={`w-6 h-6 text-${color}-600`} />
            </div>
            <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-1">{title}</h3>
                <p className="text-sm text-gray-600">{desc}</p>
            </div>
        </div>
    </button>
);

// ============================================================================
// HOMEPAGE COMPONENT
// ============================================================================
const HomePage = ({ onNavigate }) => {
    const features = [
        {
            icon: Microscope,
            title: 'Visual Intelligence',
            desc: 'AI-powered analysis of battery images and visual data',
            action: 'visual',
            color: 'blue'
        },
        {
            icon: Terminal,
            title: 'Log Analysis',
            desc: 'Parse and analyze BMS error logs with semantic understanding',
            action: 'logs',
            color: 'green'
        },
        {
            icon: Activity,
            title: 'Simulation Lab',
            desc: 'Predictive modeling and aging analysis for battery systems',
            action: 'sim',
            color: 'purple'
        },
        {
            icon: Monitor,
            title: 'Fleet Monitor',
            desc: 'Real-time monitoring and management of battery fleets',
            action: 'fleet',
            color: 'orange'
        },
        {
            icon: Layers,
            title: 'PCB Manufacturing',
            desc: 'Design and analysis tools for battery management PCBs',
            action: 'pcb',
            color: 'red'
        }
    ];

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="text-center">
                <h1 className="text-3xl font-bold text-gray-900 mb-2">BatteryForge AI</h1>
                <p className="text-lg text-gray-600">Advanced Battery Intelligence Platform</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard icon={Battery} label="Active Systems" value="1,247" color="blue" />
                <StatCard icon={Gauge} label="Health Score" value="94.2%" color="green" />
                <StatCard icon={TrendingUp} label="Efficiency" value="98.7%" color="purple" />
            </div>

            {/* Features Grid */}
            <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Available Tools</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {features.map((feature, index) => (
                        <FeatureCard
                            key={index}
                            icon={feature.icon}
                            title={feature.title}
                            desc={feature.desc}
                            onClick={() => onNavigate(feature.action)}
                            color={feature.color}
                        />
                    ))}
                </div>
            </div>

            {/* System Status */}
            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-sm text-gray-600">All Systems Operational</span>
                    </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                        <div className="text-2xl font-bold text-gray-900">99.9%</div>
                        <div className="text-xs text-gray-600">Uptime</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-gray-900">24ms</div>
                        <div className="text-xs text-gray-600">Response Time</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-gray-900">1.2M</div>
                        <div className="text-xs text-gray-600">Data Points</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-gray-900">47</div>
                        <div className="text-xs text-gray-600">Active Models</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
