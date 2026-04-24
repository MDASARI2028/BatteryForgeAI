import React, { useState, useEffect } from 'react';
import { AlertTriangle, Lock } from 'lucide-react';

function DisclaimerModal({ onAccept }) {
  const [show, setShow] = useState(true);

  useEffect(() => {
    // Check if user has already accepted the disclaimer
    const accepted = localStorage.getItem('disclaimerAccepted');
    if (accepted) {
      setShow(false);
      onAccept && onAccept();
    }
  }, [onAccept]);

  const handleAccept = () => {
    localStorage.setItem('disclaimerAccepted', 'true');
    setShow(false);
    onAccept && onAccept();
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-[9999] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-gradient-to-b from-slate-900 to-slate-950 border-2 border-indigo-500/50 rounded-2xl max-w-2xl w-full shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-indigo-800 p-8 border-b border-indigo-500/50">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-indigo-500/20 rounded-lg border border-indigo-400/50">
              <AlertTriangle className="w-8 h-8 text-indigo-300" />
            </div>
            <h1 className="text-3xl font-black text-white">
              ⚡ IMPORTANT NOTICE
            </h1>
          </div>
          <p className="text-indigo-100 font-semibold">Proprietary Access Agreement</p>
        </div>

        {/* Content */}
        <div className="p-8 space-y-6">
          {/* Company Notice */}
          <div className="bg-indigo-500/10 border-l-4 border-indigo-500 p-4 rounded">
            <p className="text-indigo-200 font-bold mb-2">Project Owner:</p>
            <p className="text-white font-black text-xl">TATA MOTORS PVT LTD</p>
            <p className="text-slate-300 text-sm mt-2">
              Advanced Battery Management & AI Intelligence Platform
            </p>
          </div>

          {/* Main Disclaimer Text */}
          <div className="space-y-4 text-slate-200 text-sm leading-relaxed">
            <div>
              <h2 className="text-indigo-400 font-bold mb-2">🔒 Proprietary Software Notice</h2>
              <p>
                This web application is proprietary software developed exclusively for <span className="font-bold text-indigo-300">Tata Motors Pvt Ltd</span>. 
                The BatteryForgeAI platform is confidential and contains trade secrets, intellectual property, 
                and technology that are protected under applicable laws.
              </p>
            </div>

            <div>
              <h2 className="text-indigo-400 font-bold mb-2">⚠️ Authorized Use Only</h2>
              <p>
                This tool is authorized for use ONLY by:
              </p>
              <ul className="list-disc list-inside ml-2 mt-2 space-y-1 text-slate-300">
                <li>Tata Motors Pvt Ltd employees and authorized contractors</li>
                <li>Individuals with explicit written permission from Tata Motors</li>
              </ul>
            </div>

            <div>
              <h2 className="text-indigo-400 font-bold mb-2">❌ Prohibited Use</h2>
              <p>
                Unauthorized access, copying, distribution, modification, or commercial use of this platform is strictly prohibited. 
                Violators will be subject to legal action and penalties as per Indian law and international regulations.
              </p>
            </div>

            <div>
              <h2 className="text-indigo-400 font-bold mb-2">📋 Regulations & Compliance</h2>
              <p>
                Use of this platform is governed by:
              </p>
              <ul className="list-disc list-inside ml-2 mt-2 space-y-1 text-slate-300">
                <li>Tata Motors Information Security Policy</li>
                <li>Data Protection & Privacy Laws</li>
                <li>Intellectual Property Rights</li>
                <li>Internal Access Controls & NDA Requirements</li>
              </ul>
            </div>
          </div>

          {/* Risk Acknowledgment */}
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg">
            <p className="text-red-300 text-sm">
              <span className="font-bold">⚠️ Disclaimer:</span> This tool is provided as-is for authorized Tata Motors operations. 
              Unauthorized use constitutes a violation of company policy and applicable law.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-slate-950 border-t border-indigo-500/30 p-6 flex gap-4">
          <button
            onClick={() => {
              alert('Access Denied: This application is for authorized Tata Motors use only.');
              // Optionally redirect or close window
            }}
            className="flex-1 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-lg transition border border-slate-700 hover:border-slate-600"
          >
            Decline
          </button>
          <button
            onClick={handleAccept}
            className="flex-1 px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-bold rounded-lg transition shadow-lg hover:shadow-indigo-500/50 flex items-center justify-center gap-2"
          >
            <Lock className="w-4 h-4" />
            I Acknowledge & Accept
          </button>
        </div>
      </div>
    </div>
  );
}

export default DisclaimerModal;
