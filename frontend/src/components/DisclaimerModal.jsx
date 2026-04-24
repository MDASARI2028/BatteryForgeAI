import React, { useState, useEffect } from 'react';
import { AlertTriangle, Lock, X } from 'lucide-react';

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

  const handleDecline = () => {
    // Show warning but allow access
    const confirmed = window.confirm(
      '⚠️ WARNING: You are accessing proprietary Tata Motors software without authorization.\n\n' +
      'This may violate company policy and applicable laws.\n\n' +
      'Are you sure you want to proceed?'
    );

    if (confirmed) {
      // Allow access but don't save to localStorage (will show again on refresh)
      setShow(false);
      onAccept && onAccept();
    }
  };

  const handleAccept = () => {
    // Save acceptance to localStorage and hide modal
    localStorage.setItem('disclaimerAccepted', 'true');
    setShow(false);
    onAccept && onAccept();
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-[9999] bg-black/50 flex items-center justify-center p-4">
      <div className="bg-white border border-gray-200 rounded-lg max-w-lg w-full shadow-lg">
        {/* Header */}
        <div className="border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <h1 className="text-lg font-semibold text-gray-900">
              Important Notice
            </h1>
          </div>
          <button
            onClick={handleDecline}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
            title="Close"
          >
            <X className="w-4 h-4 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          {/* Company Notice */}
          <div className="bg-blue-50 border border-blue-200 p-3 rounded">
            <p className="text-blue-800 font-medium text-sm">TATA MOTORS PVT LTD</p>
            <p className="text-blue-600 text-xs mt-1">
              Proprietary Battery Management Platform
            </p>
          </div>

          {/* Main Disclaimer Text */}
          <div className="space-y-3 text-sm text-gray-700">
            <p>
              <strong>This is proprietary software</strong> developed exclusively for Tata Motors Pvt Ltd.
              It contains confidential information and trade secrets.
            </p>

            <p>
              <strong>Authorized use only:</strong> This tool is for Tata Motors employees and authorized contractors only.
            </p>

            <p className="text-red-600 font-medium">
              Unauthorized access, copying, or use is prohibited and may result in legal action.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 flex gap-3">
          <button
            onClick={handleDecline}
            className="flex-1 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded transition-colors"
          >
            Proceed Anyway
          </button>
          <button
            onClick={handleAccept}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded transition-colors"
          >
            I Accept
          </button>
        </div>
      </div>
    </div>
  );
}

export default DisclaimerModal;
