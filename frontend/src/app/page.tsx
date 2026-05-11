"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Image from "next/image";
import dynamic from "next/dynamic";

export interface DetectionResult {
  image: string;
  cam_key: string;
  folder_path?: string;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  conf: number;
  x: number;
  y: number;
  z: number;
  clustered?: boolean;
  verified?: boolean;
}

export interface Session {
  id: string;
  folderPath: string;
  lasFolderPath: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'cancelled';
  progress: number;
  cameraProgress: {Camera1: number, Camera2: number, Camera3: number, Camera4: number};
  terminalLog: string[];
  resultsData: DetectionResult[];
  trashBin: DetectionResult[];
  processDuration: string;
  startTime: number;
}

// Dynamically import MapView to prevent SSR issues with Leaflet
const MapView = dynamic(() => import("./MapView"), { ssr: false, loading: () => <div className="h-full w-full bg-gray-100 animate-pulse flex items-center justify-center text-gray-400 font-mono text-xs">Loading OpenStreetMap...</div> });

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  
  // View mode tab for the verification dashboard: 'active' or 'trash'
  const [dashboardViewMode, setDashboardViewMode] = useState<'active' | 'trash'>('active');

  // Form State
  const [folderPath, setFolderPath] = useState("");
  const [lasFolderPath, setLasFolderPath] = useState("");
  const [minConfidence, setMinConfidence] = useState(75);
  const [parallaxRadius, setParallaxRadius] = useState(5.00);
  const [batchSize, setBatchSize] = useState(24);
  const [executionMode, setExecutionMode] = useState<'parallel' | 'sequential'>('sequential');
  const [isRulesExpanded, setIsRulesExpanded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Lightbox State
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [zoomOrigin, setZoomOrigin] = useState({ x: 50, y: 50 });
  const modalWrapperRef = useRef<HTMLDivElement>(null);
  const zoomImageRef = useRef<HTMLDivElement>(null);

  const activeSession = sessions.find(s => s.id === activeSessionId) || null;
  const currentImageArray = activeSession ? (dashboardViewMode === 'active' ? activeSession.resultsData : activeSession.trashBin) : [];

  // Poll running sessions
  useEffect(() => {
    const handle = setInterval(async () => {
      setSessions(prevSessions => {
        const activeSessions = prevSessions.filter(s => s.status === 'running' || s.status === 'pending');
        if (activeSessions.length === 0) return prevSessions;
        
        activeSessions.forEach(async (rs) => {
          try {
            const statusRes = await fetch(`http://localhost:8000/api/status?session_id=${rs.id}`);
            const data = await statusRes.json();
            
            setSessions(currentSessions => currentSessions.map(cs => {
              if (cs.id !== rs.id) return cs;
              
              const updatedSession = {
                ...cs,
                progress: data.progress,
                cameraProgress: data.camera_progress || cs.cameraProgress,
                terminalLog: data.logs || cs.terminalLog
              };
              
              if (data.is_running && cs.status === 'pending') {
                  updatedSession.status = 'running';
              }
              
              if (data.is_cancelled) {
                  updatedSession.status = 'cancelled';
              }
              
              if (!data.is_running && data.results_ready && (cs.status === 'running' || cs.status === 'pending')) {
                updatedSession.status = 'completed';
                fetchResults(rs.id, cs.startTime);
              }
              return updatedSession;
            }));
          } catch (e) {
            console.error("Polling error for", rs.id, e);
          }
        });
        return prevSessions; 
      });
    }, 1000);
    return () => clearInterval(handle);
  }, []);

  const fetchResults = async (sessionId: string, startTime: number) => {
    try {
      const resData = await fetch(`http://localhost:8000/api/results?session_id=${sessionId}`);
      const finalResults = await resData.json();

      const durationMs = Date.now() - startTime;
      const m = Math.floor(durationMs / 60000);
      const s = Math.floor((durationMs % 60000) / 1000);
      const formattedDuration = `${m}m ${s}s`;

      if (!Array.isArray(finalResults) || finalResults.length === 0) {
        // Automatically hide sessions with 0 results silently
        setSessions(current => current.filter(cs => cs.id !== sessionId));
        if (activeSessionId === sessionId) setActiveSessionId(null);
      } else {
        setSessions(current => current.map(cs => {
          if (cs.id !== sessionId) return cs;
          return {
            ...cs,
            processDuration: formattedDuration,
            resultsData: finalResults,
            trashBin: []
          };
        }));
      }
    } catch (e) {
       console.error("Error fetching final results", e);
    }
  };

  const handleStartAnalysis = async () => {
    if (!folderPath || !lasFolderPath) {
      alert("Please enter both valid Master Directory & LAS directory paths.");
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_path: folderPath,
          las_folder_path: lasFolderPath,
          min_confidence: minConfidence,
          cluster_radius: parallaxRadius,
          batch_size: batchSize,
          execution_mode: executionMode
        })
      });

      if (!res.ok) throw new Error("API not reachable");

      const data = await res.json();
      
      if (data.status === 'error') {
         alert(data.message);
         setIsSubmitting(false);
         return;
      }
      
      const newSessions: Session[] = data.sessions.map((rec: any) => ({
        id: rec.session_id,
        folderPath: rec.path,
        lasFolderPath,
        status: executionMode === 'sequential' ? 'pending' : 'running',
        progress: 0,
        cameraProgress: {Camera1: 0, Camera2: 0, Camera3: 0, Camera4: 0},
        terminalLog: [executionMode === 'sequential' ? "Item queued in sequential batch mode..." : "Initializing API connection to local node..."],
        resultsData: [],
        trashBin: [],
        processDuration: "0m 0s",
        startTime: Date.now()
      }));

      setSessions(prev => [...newSessions, ...prev]);
      setFolderPath("");
      setLasFolderPath("");
    } catch (e) {
      console.error("Failed to start analysis:", e);
      alert("ERROR: Could not connect to Gauss Python Node on Port 8000.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelAnalysis = async (sessionId: string) => {
    try {
      await fetch(`http://localhost:8000/api/cancel?session_id=${sessionId}`, { method: "POST" });
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, status: 'cancelled' } : s));
    } catch (e) {
      console.error(e);
    }
  };

  const handleSoftDelete = (image_name: string, cam_key: string) => {
    if (!activeSessionId) return;
    
    setSessions(prev => prev.map(s => {
      if (s.id !== activeSessionId) return s;
      const targetItem = s.resultsData.find(d => d.image === image_name && d.cam_key === cam_key);
      if (!targetItem) return s;
      
      return {
        ...s,
        resultsData: s.resultsData.filter(d => !(d.image === image_name && d.cam_key === cam_key)),
        trashBin: [...s.trashBin, targetItem]
      };
    }));
  };

  const handleRestore = (image_name: string, cam_key: string) => {
    if (!activeSessionId) return;
    
    setSessions(prev => prev.map(s => {
      if (s.id !== activeSessionId) return s;
      const targetItem = s.trashBin.find(d => d.image === image_name && d.cam_key === cam_key);
      if (!targetItem) return s;
      
      return {
        ...s,
        trashBin: s.trashBin.filter(d => !(d.image === image_name && d.cam_key === cam_key)),
        resultsData: [...s.resultsData, targetItem]
      };
    }));
  };

  const handleGenerateFinalExport = async () => {
    if (!activeSessionId || !activeSession || !allVerifiedAndClassified) return;
    
    try {
      const payload = activeSession.resultsData;

      // Backend generation STRICTLY uses activeSession.resultsData (excluding trashBin logic naturally!)
      const res = await fetch("http://localhost:8000/api/generate_final_export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: activeSessionId, results: payload })
      });
      if (!res.ok) throw new Error("Failed to generate export");
      
      window.location.href = `http://localhost:8000/api/download_shapefile?session_id=${activeSessionId}`;
    } catch(e) {
      console.error(e);
      alert("Error exporting data");
    }
  };

  // Lightbox Handlers
  const handleOpenLightbox = (index: number) => { setSelectedIndex(index); setZoomLevel(1); setZoomOrigin({ x: 50, y: 50 }); };
  const handleCloseLightbox = () => { setSelectedIndex(null); setZoomLevel(1); setZoomOrigin({ x: 50, y: 50 }); };
  const handlePrevImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentImageArray.length === 0) return;
    setSelectedIndex(prev => (prev !== null && prev > 0) ? prev - 1 : currentImageArray.length - 1);
    setZoomLevel(1);
    setZoomOrigin({ x: 50, y: 50 });
  };
  const handleNextImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentImageArray.length === 0) return;
    setSelectedIndex(prev => (prev !== null && prev < currentImageArray.length - 1) ? prev + 1 : 0);
    setZoomLevel(1);
    setZoomOrigin({ x: 50, y: 50 });
  };
  const handleZoomIn = (e: React.MouseEvent) => { e.stopPropagation(); setZoomLevel(prev => Math.min(prev + 0.5, 4)); };
  const handleZoomOut = (e: React.MouseEvent | KeyboardEvent) => {
    if ('stopPropagation' in e) e.stopPropagation();
    setZoomLevel(prev => Math.max(prev - 0.5, 0.5));
  };

  useEffect(() => {
    if (selectedIndex !== null) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => { document.body.style.overflow = 'unset'; };
  }, [selectedIndex]);

  useEffect(() => {
    const wrapper = modalWrapperRef.current;
    if (!wrapper) return;

    const handleNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      
      if (zoomImageRef.current) {
        const rect = zoomImageRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setZoomOrigin({ x, y });
      }

      if (e.deltaY < 0) {
        setZoomLevel(prev => Math.min(prev + 0.2, 5));
      } else {
        setZoomLevel(prev => Math.max(prev - 0.2, 0.5));
      }
    };

    wrapper.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => wrapper.removeEventListener('wheel', handleNativeWheel);
  }, [selectedIndex]);

  const handleToggleVerify = useCallback(() => {
    if (selectedIndex === null || !activeSessionId || dashboardViewMode === 'trash') return;
    setSessions(prev => prev.map(s => {
      if (s.id !== activeSessionId) return s;
      const copy = [...s.resultsData];
      copy[selectedIndex] = { ...copy[selectedIndex], verified: !copy[selectedIndex].verified };
      return { ...s, resultsData: copy };
    }));
  }, [selectedIndex, activeSessionId, dashboardViewMode]);



  const handleDeleteCurrent = useCallback(() => {
    if (selectedIndex === null || !activeSession) return;
    const item = currentImageArray[selectedIndex];
    
    if (dashboardViewMode === 'active') {
        handleSoftDelete(item.image, item.cam_key);
    } else {
        handleRestore(item.image, item.cam_key);
    }
    
    if (currentImageArray.length <= 1) {
      handleCloseLightbox();
    } else if (selectedIndex >= currentImageArray.length - 1) {
      setSelectedIndex(selectedIndex - 1);
    }
  }, [selectedIndex, activeSession, dashboardViewMode, currentImageArray]);

  useEffect(() => {
    if (selectedIndex === null || !activeSession) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") handleNextImage(e as any);
      else if (e.key === "ArrowLeft") handlePrevImage(e as any);
      else if (e.key === "Escape") handleCloseLightbox();
      else if (e.key === " " && dashboardViewMode === 'active') { e.preventDefault(); handleToggleVerify(); }
      else if (e.key.toLowerCase() === "x") { e.preventDefault(); handleDeleteCurrent(); }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIndex, activeSession, handleToggleVerify, handleDeleteCurrent, dashboardViewMode]);

  const allVerifiedAndClassified = activeSession?.resultsData.length ? activeSession.resultsData.every(r => r.verified) : false;

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm relative z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image src="/gauss_logo.png" alt="Gauss Logo" width={140} height={40} className="object-contain" priority/>
            <div className="h-8 w-px bg-gray-300 hidden sm:block"></div>
            <h1 className="text-xl font-bold text-gray-700 tracking-tight hidden sm:block">
              Detector Camine
            </h1>
          </div>
          <div className="text-sm border border-brand-primary text-brand-primary px-3 py-1 rounded-full font-semibold uppercase tracking-wider">
            Enterprise Node Active
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full relative z-10">
        {!activeSessionId ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden h-fit">
              <div className="p-6 border-b border-gray-100">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Start Master Directory Batch Processing</h2>
                
                <div className="flex flex-col gap-4">
                  <div className="bg-gray-50 flex flex-col gap-2 p-4 rounded-lg border border-gray-200">
                    <label className="text-sm font-semibold text-gray-700">Master Trident Directory Path</label>
                    <input 
                      type="text" 
                      value={folderPath}
                      onChange={(e) => setFolderPath(e.target.value)}
                      placeholder="e.g. \\srvNAS5\DEER2024\Constanta_Zone1"
                      className="w-full px-3 py-2 border border-gray-300 rounded focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-sm font-mono shadow-inner"
                    />
                    <p className="text-xs text-gray-500 mt-1">Provide the parent folder. The engine will loop through subfolders inside it to spawn isolated session queues.</p>
                  </div>
                  
                  <div className="bg-gray-50 flex flex-col gap-2 p-4 rounded-lg border border-gray-200">
                    <label className="text-sm font-semibold text-gray-700">LiDAR Point Cloud (.las) Directory</label>
                    <input 
                      type="text" 
                      value={lasFolderPath}
                      onChange={(e) => setLasFolderPath(e.target.value)}
                      placeholder="e.g. C:\Data\LiDAR_Blocks\..."
                      className="w-full px-3 py-2 border border-blue-200 bg-blue-50/30 rounded focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-sm font-mono shadow-inner"
                    />
                  </div>
                </div>
              </div>

              <div className="border-b border-gray-100">
                <button 
                  onClick={() => setIsRulesExpanded(!isRulesExpanded)}
                  className="w-full px-6 py-4 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors outline-none"
                >
                  <span className="text-sm font-semibold text-gray-700">Advanced Rules</span>
                  <svg className={`w-5 h-5 text-gray-500 transform transition-transform ${isRulesExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {isRulesExpanded && (
                  <div className="px-6 py-4 space-y-4 bg-white animate-fade-in">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Minimum Confidence (%)</span>
                      <input 
                        type="number" 
                        value={minConfidence}
                        onChange={(e) => setMinConfidence(parseInt(e.target.value) || 0)}
                        className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-center font-mono" 
                        min="0" max="100" step="1"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Raycast Intersection Auth Radius (m)</span>
                      <input 
                        type="number" 
                        value={parallaxRadius.toFixed(2)}
                        onChange={(e) => setParallaxRadius(parseFloat(e.target.value))}
                        className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-center font-mono" 
                        step="0.01"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">System GPU Batch Size</span>
                      <input 
                        type="number" 
                        value={batchSize}
                        onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                        className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-center font-mono" 
                        min="1" max="512" step="1"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="p-6 bg-gray-50 flex flex-col gap-4">
                <div className="flex bg-white rounded-lg p-1 border border-gray-200">
                  <button 
                    type="button"
                    onClick={() => setExecutionMode('parallel')}
                    className={`flex-1 py-2 text-sm font-bold rounded shadow-sm transition-colors ${executionMode === 'parallel' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 'text-gray-500 hover:bg-gray-50'}`}
                  >
                    Analyze All (Parallel)
                  </button>
                  <button 
                    type="button"
                    onClick={() => setExecutionMode('sequential')}
                    className={`flex-1 py-2 text-sm font-bold rounded shadow-sm transition-colors ${executionMode === 'sequential' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 'text-gray-500 hover:bg-gray-50'}`}
                  >
                    One-by-One (Sequential)
                  </button>
                </div>
                
                <button
                  onClick={handleStartAnalysis}
                  disabled={isSubmitting}
                  className={`w-full py-4 px-6 rounded-lg font-bold uppercase tracking-wide transition-all shadow-md flex items-center justify-center gap-2 ${
                    isSubmitting
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none' 
                      : 'bg-brand-primary text-white hover:bg-brand-secondary hover:shadow-lg hover:-translate-y-0.5'
                  }`}
                >
                  Launch Batch Analysis Process
                </button>
              </div>
            </div>
            
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col h-full min-h-[500px]">
              <div className="p-4 border-b border-gray-100 bg-gray-50">
                <h2 className="text-lg font-bold text-gray-900">Batch Job Dashboard ({sessions.length})</h2>
              </div>
              <div className="p-4 flex-1 overflow-y-auto space-y-4 bg-gray-50">
                 {sessions.length === 0 ? (
                   <div className="h-full flex flex-col items-center justify-center text-gray-400">
                     No active or past sessions.
                   </div>
                 ) : (
                   sessions.map(s => {
                     const folderName = s.folderPath.split(/[\\/]/).pop() || "Unknown Data";
                     return (
                       <div key={s.id} className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 flex flex-col gap-3 transition hover:shadow-md">
                         <div className="flex justify-between items-start">
                           <div className="flex flex-col">
                             <span className="font-bold text-gray-800 break-all text-sm">{folderName}</span>
                             <span className="text-[10px] text-gray-400 font-mono mt-0.5">uuid: {s.id.split('-')[0]}</span>
                           </div>
                           <div className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${s.status === 'running' ? 'bg-blue-100 text-blue-700' : s.status === 'completed' ? 'bg-green-100 text-green-700' : s.status === 'pending' ? 'bg-gray-200 text-gray-700' : 'bg-red-100 text-red-700'}`}>
                             {s.status}
                           </div>
                         </div>
                         
                         {(s.status === 'running' || s.status === 'pending') && (
                           <>
                             <div className="flex items-center gap-3">
                                <div className="flex-1 bg-gray-100 rounded-full h-2.5 overflow-hidden border border-gray-200 shadow-inner">
                                   <div className={`h-2.5 rounded-full transition-all duration-300 relative ${s.status === 'pending' ? 'bg-gray-300' : 'bg-blue-500'}`} style={{width: `${s.progress}%`}}>
                                     {s.status === 'running' && <div className="absolute inset-0 bg-white/20 animate-pulse"></div>}
                                   </div>
                                </div>
                                <span className="font-mono font-bold text-sm text-blue-700 min-w-[36px]">{s.progress}%</span>
                             </div>
                             <div className="flex justify-end gap-2">
                               <button onClick={() => handleCancelAnalysis(s.id)} className="text-xs text-red-500 hover:text-red-700 transition">Halt Process</button>
                             </div>
                           </>
                         )}
                         
                         {s.status === 'completed' && (
                           <div className="border-t border-gray-100 pt-3 mt-1 flex justify-between items-center">
                             <span className="text-xs text-gray-500">Found {s.resultsData.length} records.</span>
                             <button 
                               onClick={() => setActiveSessionId(s.id)}
                               className="bg-brand-primary text-white px-4 py-1.5 rounded text-sm font-bold shadow hover:bg-brand-secondary transition-colors"
                             >
                               Review Map Layer
                             </button>
                           </div>
                         )}
                       </div>
                     );
                   })
                 )}
              </div>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in h-full flex flex-col gap-6 w-full fade-in">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-4 bg-white p-4 rounded-xl shadow-sm border border-gray-200">
              <div>
                <h2 className="text-2xl font-black text-gray-900 tracking-tight flex items-center gap-3">
                  <button onClick={() => { setActiveSessionId(null); setDashboardViewMode('active'); }} className="text-gray-400 hover:text-brand-primary transition-colors">
                     <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
                  </button>
                  Verification Console
                </h2>
                <div className="flex flex-col sm:flex-row items-baseline gap-3 mt-1 ml-9">
                  <p className="text-sm text-gray-500">Volumetric matches from <span className="font-mono font-semibold">{activeSession?.folderPath.split(/[\\/]/).pop()}</span></p>
                </div>
              </div>

              {/* View Toggle */}
              <div className="bg-gray-100 p-1 rounded-lg border border-gray-200 flex ml-4 shadow-inner">
                 <button 
                   onClick={() => setDashboardViewMode('active')} 
                   className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${dashboardViewMode === 'active' ? 'bg-white text-brand-primary shadow' : 'text-gray-500 hover:text-gray-700'}`}
                 >
                   Active Queue ({activeSession?.resultsData.length || 0})
                 </button>
                 <button 
                   onClick={() => setDashboardViewMode('trash')} 
                   className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all flex items-center gap-1.5 ${dashboardViewMode === 'trash' ? 'bg-red-500 text-white shadow' : 'text-gray-500 hover:text-red-500'}`}
                 >
                   <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                   Trash Bin ({activeSession?.trashBin.length || 0})
                 </button>
              </div>

              <button 
                onClick={() => { setActiveSessionId(null); setDashboardViewMode('active'); }}
                className="text-sm border-2 border-brand-primary text-brand-primary hover:bg-brand-primary hover:text-white px-4 py-2 rounded-full font-semibold tracking-wide flex items-center gap-2 transition-colors ml-auto sm:ml-0"
              >
                Exit Console
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className={`lg:col-span-2 rounded-xl shadow-sm border p-6 flex flex-col overflow-hidden max-h-[75vh] transition-colors ${dashboardViewMode === 'trash' ? 'bg-red-50/30 border-red-200' : 'bg-white border-gray-200'}`}>
                <div className={`flex justify-between items-center mb-4 border-b pb-3 ${dashboardViewMode === 'trash' ? 'border-red-100' : 'border-gray-100'}`}>
                   <h3 className={`text-sm font-bold uppercase tracking-wider ${dashboardViewMode === 'trash' ? 'text-red-800' : 'text-gray-900'}`}>
                      {dashboardViewMode === 'trash' ? 'Discarded Objects (Soft Delete)' : 'Detected Object Postages'}
                   </h3>
                   <span className="text-xs bg-white text-gray-600 px-2 py-1 rounded font-mono border shadow-sm">Displayed: {currentImageArray.length}</span>
                </div>
                
                <div className="flex-1 overflow-y-auto min-h-0 pr-2 pb-4 custom-scrollbar">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {!currentImageArray || currentImageArray.length === 0 ? (
                    <div className="col-span-full py-12 flex flex-col items-center justify-center text-gray-400">
                       <p>{dashboardViewMode === 'trash' ? 'No items in the Trash Bin.' : 'No active detections.'}</p>
                    </div>
                  ) : (
                    currentImageArray.map((item, idx) => (
                      <div 
                        key={`${item.cam_key}_${item.image}_${idx}`} 
                        className={`border rounded-xl overflow-hidden relative group shadow-sm transition-all hover:shadow-md cursor-pointer ${dashboardViewMode === 'trash' ? 'border-red-300 opacity-80' : (item.clustered ? 'border-yellow-400 ring-1 ring-yellow-400/50 shadow-yellow-100' : 'border-gray-200')}`}
                        onClick={() => handleOpenLightbox(idx)}
                      >
                        <div className={`absolute top-2 left-2 text-white text-[10px] font-bold px-2 py-0.5 rounded shadow z-10 flex items-center gap-1 ${item.clustered ? 'bg-yellow-500' : (item.conf >= 0.85 ? 'bg-green-500' : 'bg-orange-500')}`}>
                          {item.clustered ? "CLUSTERED" : `${(item.conf * 100).toFixed(0)}% CONF`}
                        </div>
                        
                        {dashboardViewMode === 'active' ? (
                          <button 
                            onClick={(e) => { e.stopPropagation(); handleSoftDelete(item.image, item.cam_key); }}
                            className="absolute top-2 right-2 bg-red-500 text-white rounded p-1.5 shadow-lg opacity-0 group-hover:opacity-100 hover:bg-red-600 transform transition-all hover:scale-110 z-20"
                            title="Move to Trash"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        ) : (
                           <button 
                            onClick={(e) => { e.stopPropagation(); handleRestore(item.image, item.cam_key); }}
                            className="absolute top-2 right-2 bg-green-500 text-white rounded px-2 py-1 shadow-lg opacity-0 group-hover:opacity-100 hover:bg-green-600 transform transition-all hover:scale-105 z-20 text-[10px] uppercase font-bold"
                            title="Restore Target"
                          >
                            Restore
                          </button>
                        )}

                        <div className="aspect-square bg-gray-100 relative overflow-hidden">
                          <Image src={`http://localhost:8000/images/${activeSessionId}/${item.cam_key}_${item.image}`} alt={item.image} fill className={`object-cover transition-transform duration-700 ease-in-out group-hover:scale-110 unoptimized ${dashboardViewMode === 'trash' ? 'grayscale' : ''}`} unoptimized/>
                        </div>
                        <div className={`p-3 text-[10px] font-mono border-t flex flex-col gap-0.5 ${dashboardViewMode === 'trash' ? 'bg-red-50 text-red-800' : (item.clustered ? 'bg-yellow-50 text-yellow-800 border-yellow-200' : 'bg-gray-50 text-gray-500 border-gray-200')}`}>
                          <span className="truncate w-full font-bold" title={item.image}>{item.image}</span>
                          <span className="opacity-70">{item.cam_key}</span>
                        </div>
                      </div>
                    ))
                  )}
                  </div>
                </div>
              </div>

              <div className="lg:col-span-1 space-y-6 flex flex-col h-full">
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-[400px]">
                  <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-4 border-b border-gray-100 pb-3 flex justify-between items-center">
                    Active Spatial Context
                  </h3>
                  <div className="flex-1 rounded-lg overflow-hidden relative shadow-inner z-0 border border-gray-200">
                     <MapView data={activeSession?.resultsData || []} />
                  </div>
                </div>

                <div className="bg-brand-primary p-6 rounded-xl shadow-lg relative overflow-hidden group">
                  <div className="absolute -top-10 -right-10 w-32 h-32 bg-white opacity-5 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-1000"></div>
                  <h3 className="text-white font-bold mb-2 text-lg">Export Deliverable</h3>
                  <p className="text-brand-light/90 text-sm mb-6 leading-relaxed">
                    Verify and classify all current {activeSession?.resultsData.length} instances to enable export. Soft-deleted items are safely ignored.
                  </p>
                  <button 
                    onClick={handleGenerateFinalExport}
                    disabled={!allVerifiedAndClassified || activeSession?.resultsData.length === 0}
                    className={`w-full py-3.5 rounded-lg font-bold text-sm tracking-widest uppercase flex items-center justify-center gap-3 transition-all ${allVerifiedAndClassified && activeSession?.resultsData.length ? 'bg-white text-brand-primary hover:shadow-xl hover:bg-gray-50 transform group-hover:-translate-y-1 cursor-pointer' : 'bg-white/30 text-white/50 cursor-not-allowed'}`}
                  >
                    Generate Shapefile
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Lightbox Modal */}
      {selectedIndex !== null && currentImageArray[selectedIndex] && (
        <div className="fixed inset-0 z-[100] bg-black/90 flex items-center justify-center p-4 sm:p-8 animate-fade-in" onClick={handleCloseLightbox}>
          <button className="absolute top-4 right-4 text-white bg-black/50 hover:bg-black/80 rounded-full p-2 z-[110] transition-colors" onClick={handleCloseLightbox}>
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>

          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur rounded-full px-6 py-3 flex items-center gap-6 z-[110] text-white shadow-xl border border-white/10" onClick={(e) => e.stopPropagation()}>
            <button onClick={handlePrevImage} className="hover:text-brand-accent transition-colors">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
            </button>
            <div className="w-px h-6 bg-white/20"></div>
            <button onClick={handleZoomOut} className="hover:text-brand-accent transition-colors">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" /></svg>
            </button>
            <span className="font-mono font-bold text-sm w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
            <button onClick={handleZoomIn} className="hover:text-brand-accent transition-colors">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>
            </button>
            <div className="w-px h-6 bg-white/20"></div>
            <button onClick={handleNextImage} className="hover:text-brand-accent transition-colors">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
            </button>
          </div>

          <div ref={modalWrapperRef} className="w-full h-full flex flex-col items-center justify-center overflow-auto custom-scrollbar" onClick={(e) => e.stopPropagation()}>
            <div ref={zoomImageRef} className={`relative transition-transform duration-200 ease-out flex-shrink-0 ${dashboardViewMode === 'trash' ? 'grayscale' : ''}`} style={{ transform: `scale(${zoomLevel})`, transformOrigin: `${zoomOrigin.x}% ${zoomOrigin.y}%` }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={`http://localhost:8000/images/${activeSessionId}/${currentImageArray[selectedIndex].cam_key}_${currentImageArray[selectedIndex].image}`} 
                alt={currentImageArray[selectedIndex].image}
                className="max-w-[90vw] max-h-[85vh] object-contain rounded shadow-2xl"
                draggable={false}
              />
            </div>
            
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-4 py-3 rounded-lg text-white font-mono text-sm border border-white/10 z-[110] flex flex-col gap-3 min-w-[220px]">
              <div>
                <span className="text-brand-accent font-bold">
                  {dashboardViewMode === 'trash' && <span className="text-red-400 mr-2">[TRASH]</span>}
                  [{selectedIndex + 1} / {currentImageArray.length}]
                </span><br/>
                File: {currentImageArray[selectedIndex].image}<br/>
                Camera: {currentImageArray[selectedIndex].cam_key}<br/>
                Conf: {(currentImageArray[selectedIndex].conf * 100).toFixed(0)}%
              </div>
              
              {dashboardViewMode === 'active' ? (
                  <>
                  <div className="flex flex-col gap-2 pt-2 border-t border-white/20">
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleToggleVerify(); }}
                      className={`py-1.5 px-3 rounded text-xs font-bold uppercase transition-colors flex justify-center items-center gap-1.5 w-full border ${currentImageArray[selectedIndex].verified ? 'bg-green-600 border-green-500 text-white' : 'bg-gray-700 hover:bg-gray-600 border-gray-500 text-gray-300'}`}
                    >
                      {currentImageArray[selectedIndex].verified ? '✓ Verified' : 'Verify (Space)'}
                    </button>
                    

                  </div>

                  <button 
                    onClick={(e) => { e.stopPropagation(); handleDeleteCurrent(); }}
                    className="mt-1 bg-red-600 hover:bg-red-500 text-white px-3 py-1.5 rounded text-xs font-bold uppercase flex items-center justify-center gap-1.5 transition-colors w-full shadow border border-red-500/30"
                  >
                    Move to Trash (x)
                  </button>
                  </>
              ) : (
                  <div className="pt-2 border-t border-white/20">
                      <button 
                        onClick={(e) => { e.stopPropagation(); handleDeleteCurrent(); }}
                        className="bg-green-600 hover:bg-green-500 text-white px-3 py-2 rounded text-sm font-bold uppercase flex items-center justify-center gap-1.5 transition-colors w-full shadow border border-green-500/30"
                      >
                        Restore Target (x)
                      </button>
                  </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
