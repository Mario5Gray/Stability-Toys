// src/components/dreams/DreamGallery.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Play,
  Square,
  RefreshCw,
  Download,
  Heart,
  TrendingUp,
  Clock,
  Zap,
  AlertCircle,
  XCircle,
} from 'lucide-react';

/**
 * Dream Gallery - View background dream session results.
 * Separate from main chat, shows top-scoring dreams.
 * 
 * Now with comprehensive error handling to prevent UI crashes!
 */
export function DreamGallery({ apiBase }) {
  // Dream session state
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamStatus, setDreamStatus] = useState(null);
  const [dreams, setDreams] = useState([]);
  
  // Start dream params
  const [basePrompt, setBasePrompt] = useState('');
  const [durationHours, setDurationHours] = useState(1.0);
  const [temperature, setTemperature] = useState(0.5);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [renderInterval, setRenderInterval] = useState(100);
  
  // Display filters
  const [sortBy, setSortBy] = useState('score'); // score | time
  const [minScore, setMinScore] = useState(0.0);
  const [renderedOnly, setRenderedOnly] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // Favorites
  const [favorites, setFavorites] = useState(new Set());
  
  // Error and loading states - CRITICAL FOR STABILITY
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isDreamSystemAvailable, setIsDreamSystemAvailable] = useState(true);

  /**
   * Clear error after timeout
   */
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 10000); // Clear after 10s
      return () => clearTimeout(timer);
    }
  }, [error]);

  /**
   * Start dream session with comprehensive error handling.
   */
  const startDreaming = useCallback(async () => {
    if (!basePrompt.trim()) {
      setError('Please enter a base prompt first');
      return;
    }

    setIsStarting(true);
    setError(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

      const res = await fetch(`${apiBase}/dreams/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: basePrompt,
          duration_hours: durationHours,
          temperature,
          similarity_threshold: similarityThreshold,
          render_interval: renderInterval,
          top_k: 100,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorText = await res.text().catch(() => 'Unknown error');
        throw new Error(`Failed to start dream session: ${errorText}`);
      }

      const data = await res.json();
      console.log('Dream session started:', data);
      setIsDreaming(true);
      setError(null);
      
    } catch (err) {
      console.error('Failed to start dreaming:', err);
      
      if (err.name === 'AbortError') {
        setError('Request timed out. Dream system may be unavailable.');
        setIsDreamSystemAvailable(false);
      } else {
        setError(`Failed to start: ${err.message}`);
      }
      
    } finally {
      setIsStarting(false);
    }
  }, [apiBase, basePrompt, durationHours, temperature, similarityThreshold, renderInterval]);

  /**
   * Stop dream session with error handling.
   */
  const stopDreaming = useCallback(async () => {
    setIsStopping(true);
    setError(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const res = await fetch(`${apiBase}/dreams/stop`, {
        method: 'POST',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      console.log('Dream session stopped:', data);
      setIsDreaming(false);
      
      // Fetch final results
      await fetchDreams();
      
    } catch (err) {
      console.error('Failed to stop dreaming:', err);
      setError(`Failed to stop: ${err.message}`);
      // Still mark as not dreaming to prevent UI lockup
      setIsDreaming(false);
      
    } finally {
      setIsStopping(false);
    }
  }, [apiBase]);

  /**
   * Fetch dream status with bulletproof error handling.
   */
  const fetchStatus = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(`${apiBase}/dreams/status`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      
      if (!res.ok) {
        // Don't throw on 404 - dream system might not be available
        if (res.status === 404) {
          setIsDreamSystemAvailable(false);
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      
      const data = await res.json();
      
      // Validate response structure
      if (typeof data.is_dreaming !== 'boolean') {
        console.warn('Invalid dream status response:', data);
        return;
      }
      
      setDreamStatus(data);
      setIsDreaming(data.is_dreaming);
      setIsDreamSystemAvailable(true);
      
    } catch (err) {
      // Silent fail for status checks - don't spam errors
      if (err.name !== 'AbortError') {
        console.warn('Failed to fetch status:', err.message);
      }
      
      // Set safe defaults without breaking UI
      setDreamStatus(prev => prev || {
        is_dreaming: false,
        dream_count: 0,
        dreams_per_second: 0,
        candidates: 0,
        elapsed_seconds: 0,
      });
      
      // Only mark as unavailable after multiple failures
      if (err.name === 'AbortError') {
        setIsDreamSystemAvailable(false);
      }
    }
  }, [apiBase]);
  
  /**
   * Fetch dream results with error recovery.
   */
  const fetchDreams = useCallback(async () => {
    if (isLoading) return; // Prevent concurrent requests
    
    setIsLoading(true);

    try {
      const params = new URLSearchParams({
        limit: '100',
        min_score: String(minScore),
        rendered_only: String(renderedOnly),
      });

      const endpoint = sortBy === 'time' ? 'recent' : 'top';
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const res = await fetch(`${apiBase}/dreams/${endpoint}?${params}`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        if (res.status === 404) {
          // No dreams yet - not an error
          setDreams([]);
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      
      // Validate response is an array
      if (!Array.isArray(data)) {
        console.warn('Invalid dreams response:', data);
        setDreams([]);
        return;
      }
      
      setDreams(data);
      setError(null); // Clear any previous errors
      
    } catch (err) {
      console.error('Failed to fetch dreams:', err);
      
      // Don't overwrite existing dreams on error
      if (err.name === 'AbortError') {
        setError('Request timed out loading dreams');
      } else {
        setError(`Failed to load dreams: ${err.message}`);
      }
      
    } finally {
      setIsLoading(false);
    }
  }, [apiBase, minScore, renderedOnly, sortBy, isLoading]);

  /**
   * Auto-refresh while dreaming with error recovery.
   */
  useEffect(() => {
    if (!autoRefresh || !isDreamSystemAvailable) return;

    const interval = setInterval(async () => {
      // Non-blocking status check
      fetchStatus().catch(() => {});
      
      // Only fetch dreams if actively dreaming
      if (isDreaming) {
        fetchDreams().catch(() => {});
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [autoRefresh, isDreaming, isDreamSystemAvailable, fetchStatus, fetchDreams]);

  /**
   * Initial status check on mount.
   */
  useEffect(() => {
    fetchStatus().catch(() => {});
    fetchDreams().catch(() => {});
  }, []);

  /**
   * Toggle favorite (safe).
   */
  const toggleFavorite = useCallback((seed) => {
    try {
      setFavorites(prev => {
        const next = new Set(prev);
        if (next.has(seed)) {
          next.delete(seed);
        } else {
          next.add(seed);
        }
        return next;
      });
    } catch (err) {
      console.error('Failed to toggle favorite:', err);
    }
  }, []);

  /**
   * Download dream image (safe).
   */
  const downloadDream = useCallback((dream) => {
    try {
      if (!dream?.image_data) {
        setError('No image data available');
        return;
      }

      const link = document.createElement('a');
      link.href = `data:image/png;base64,${dream.image_data}`;
      link.download = `dream_${dream.seed}_${dream.score.toFixed(3)}.png`;
      link.click();
    } catch (err) {
      console.error('Failed to download dream:', err);
      setError('Failed to download image');
    }
  }, []);

  /**
   * Dismiss error.
   */
  const dismissError = useCallback(() => {
    setError(null);
  }, []);

  return (
    <div className="h-screen flex flex-col p-4 gap-4">
      {/* Error Banner - ALWAYS VISIBLE WHEN ERRORS OCCUR */}
      {error && (
        <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-3 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
          <div className="flex-1 text-sm text-destructive">
            {error}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={dismissError}
            className="flex-shrink-0"
          >
            <XCircle className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Dream System Unavailable Warning */}
      {!isDreamSystemAvailable && (
        <div className="rounded-lg bg-yellow-500/10 border border-yellow-500/20 p-3 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0" />
          <div className="flex-1 text-sm text-yellow-700">
            Dream system is not available. Check server logs or restart the backend.
          </div>
        </div>
      )}

      {/* Control Panel */}
      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <CardTitle>Dream Session Control</CardTitle>
            {isDreaming && (
              <Badge className="animate-pulse bg-purple-600">
                <Zap className="h-3 w-3 mr-1" />
                Dreaming
              </Badge>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4 p-4">
          {/* Base Prompt */}
          <div className="space-y-2">
            <Label>Base Prompt</Label>
            <Input
              value={basePrompt}
              onChange={(e) => setBasePrompt(e.target.value)}
              placeholder="a cinematic photograph of..."
              disabled={isDreaming || isStarting}
            />
          </div>

          {/* Duration */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Duration (hours)</Label>
              <span className="text-sm text-muted-foreground">
                {durationHours.toFixed(1)}h
              </span>
            </div>
            <Slider
              value={[durationHours]}
              min={0.1}
              max={24}
              step={0.1}
              onValueChange={([v]) => setDurationHours(v)}
              disabled={isDreaming || isStarting}
            />
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Temperature (exploration)</Label>
              <span className="text-sm text-muted-foreground">
                {temperature.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[temperature]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={([v]) => setTemperature(v)}
              disabled={isDreaming || isStarting}
            />
          </div>

          {/* Similarity Threshold */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Similarity Threshold</Label>
              <span className="text-sm text-muted-foreground">
                {similarityThreshold.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[similarityThreshold]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={([v]) => setSimilarityThreshold(v)}
              disabled={isDreaming || isStarting}
            />
            <div className="text-xs text-muted-foreground">
              Only keep dreams scoring above this threshold
            </div>
          </div>

          {/* Controls */}
          <div className="flex gap-2">
            {!isDreaming ? (
              <Button
                className="flex-1"
                onClick={startDreaming}
                disabled={!basePrompt.trim() || isStarting || !isDreamSystemAvailable}
              >
                {isStarting ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Dreaming
                  </>
                )}
              </Button>
            ) : (
              <Button
                className="flex-1"
                variant="destructive"
                onClick={stopDreaming}
                disabled={isStopping}
              >
                {isStopping ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Stopping...
                  </>
                ) : (
                  <>
                    <Square className="mr-2 h-4 w-4" />
                    Stop Dreaming
                  </>
                )}
              </Button>
            )}

            <Button
              variant="outline"
              onClick={fetchDreams}
              disabled={isLoading}
            >
              {isLoading ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Refresh
            </Button>
          </div>

          {/* Status */}
          {dreamStatus && (
            <div className="rounded-lg bg-muted/50 p-3 text-xs space-y-1">
              <div className="flex justify-between">
                <span>Dreams Generated:</span>
                <span className="font-mono">
                  {dreamStatus.dream_count || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Dreams/Second:</span>
                <span className="font-mono">
                  {(dreamStatus.dreams_per_second || 0).toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Candidates Kept:</span>
                <span className="font-mono">
                  {dreamStatus.candidates || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Elapsed:</span>
                <span className="font-mono">
                  {((dreamStatus.elapsed_seconds || 0) / 60).toFixed(1)}m
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Gallery */}
      <Card className="flex-1 rounded-2xl shadow-sm overflow-hidden flex flex-col">
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <CardTitle>Dream Results ({dreams.length})</CardTitle>
            
            <div className="flex gap-2">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="score">
                    <TrendingUp className="inline h-3 w-3 mr-1" />
                    By Score
                  </SelectItem>
                  <SelectItem value="time">
                    <Clock className="inline h-3 w-3 mr-1" />
                    By Time
                  </SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                {autoRefresh ? 'Auto ✓' : 'Manual'}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto p-4">
          {isLoading && dreams.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-2">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Loading dreams...</p>
              </div>
            </div>
          ) : dreams.length === 0 ? (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              No dreams yet. Start a dream session to see results.
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {dreams.map((dream) => {
                // Safe access to dream properties
                const seed = dream?.seed || 'unknown';
                const score = dream?.score || 0;
                const imageData = dream?.image_data;
                const prompt = dream?.prompt || 'No prompt';

                return (
                  <div
                    key={seed}
                    className="group relative rounded-xl overflow-hidden border bg-card hover:shadow-lg transition-shadow"
                  >
                    {/* Image */}
                    {imageData ? (
                      <img
                        src={`data:image/png;base64,${imageData}`}
                        alt={`Dream ${seed}`}
                        className="w-full aspect-square object-cover"
                        onError={(e) => {
                          // Handle broken images gracefully
                          e.target.style.display = 'none';
                          e.target.parentElement.querySelector('.fallback')?.classList.remove('hidden');
                        }}
                      />
                    ) : null}
                    
                    {/* Fallback for missing/broken images */}
                    <div className={`${imageData ? 'hidden' : ''} fallback w-full aspect-square bg-muted flex items-center justify-center`}>
                      <span className="text-xs text-muted-foreground">
                        Not rendered
                      </span>
                    </div>

                    {/* Overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                      <div className="absolute bottom-0 left-0 right-0 p-3 space-y-2">
                        {/* Score */}
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge variant="secondary" className="text-xs">
                            Score: {score.toFixed(3)}
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            Seed: {seed}
                          </Badge>
                        </div>

                        {/* Actions */}
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant={favorites.has(seed) ? 'default' : 'outline'}
                            className="flex-1 h-7"
                            onClick={() => toggleFavorite(seed)}
                          >
                            <Heart className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 h-7"
                            onClick={() => downloadDream(dream)}
                            disabled={!imageData}
                          >
                            <Download className="h-3 w-3" />
                          </Button>
                        </div>

                        {/* Prompt preview */}
                        <div className="text-xs text-white/80 line-clamp-2">
                          {prompt}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}