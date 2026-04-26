import React, { useState } from 'react';
import { Microscope, Camera, Video, Activity } from 'lucide-react';
import UploadZone from './UploadZone';
import AnalysisResult from './AnalysisResult';
import VisualScout from './VisualScout';

const VisualIntelligence = () => {
    const [activeTab, setActiveTab] = useState('static'); // 'static' or 'live'
    const [defectResult, setDefectResult] = useState(null);

    return (
        <div className="animate-in fade-in zoom-in-95 duration-500 h-full flex flex-col px-1 sm:px-0">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 md:mb-8 gap-4 md:gap-0">
                <div className="flex items-center gap-3 md:gap-4">
                    <div className="p-2.5 md:p-3 bg-slate-800/50 rounded-xl border border-white/10 text-indigo-400 shadow-lg shrink-0">
                        <Microscope className="w-5 h-5 md:w-6 md:h-6" />
                    </div>
                    <div>
                        <h2 className="text-xl md:text-2xl font-bold text-white tracking-tight">Visual Intelligence</h2>
                        <p className="text-[11px] md:text-sm text-slate-400">Comprehensive Defect Detection & Live Analysis</p>
                    </div>
                </div>

                {/* Sub-navigation Tabs */}
                <div className="flex bg-slate-900/50 p-1 rounded-lg border border-white/5 self-start md:self-auto">
                    <button
                        onClick={() => setActiveTab('static')}
                        className={`flex items-center gap-2 px-3 md:px-4 py-1.5 md:py-2 rounded-md text-xs md:text-sm font-medium transition-all ${activeTab === 'static'
                                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
                                : 'text-slate-400 hover:text-white hover:bg-slate-800'
                            }`}
                    >
                        <Camera className="w-3.5 h-3.5 md:w-4 md:h-4" />
                        Static
                    </button>
                    <button
                        onClick={() => setActiveTab('live')}
                        className={`flex items-center gap-2 px-3 md:px-4 py-1.5 md:py-2 rounded-md text-xs md:text-sm font-medium transition-all ${activeTab === 'live'
                                ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                                : 'text-slate-400 hover:text-white hover:bg-slate-800'
                            }`}
                    >
                        <Video className="w-3.5 h-3.5 md:w-4 md:h-4" />
                        Live Scout
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 min-h-0">
                {activeTab === 'static' ? (
                    <div className="bg-slate-900/40 backdrop-blur-md rounded-2xl border border-white/5 p-4 md:p-6 shadow-2xl h-full overflow-y-auto">
                        <div className="mb-4 md:mb-6">
                            <h3 className="text-base md:text-lg font-semibold text-white mb-1 md:mb-2">Static Image Analysis</h3>
                            <p className="text-slate-400 text-xs md:text-sm">Upload high-resolution images of battery cells for defect segmentation.</p>
                        </div>
                        <UploadZone onResult={setDefectResult} />
                        <AnalysisResult data={defectResult} type="defect" />
                    </div>
                ) : (
                    <div className="h-full">
                        <VisualScout />
                    </div>
                )}
            </div>
        </div>
    );
};

export default VisualIntelligence;
