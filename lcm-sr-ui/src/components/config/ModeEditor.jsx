import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/select';
import { Slider } from '../ui/slider';
import { Badge } from '../ui/badge';
import { Trash2, Plus, Save, RefreshCw, Star } from 'lucide-react';
import { createApiClient, createApiConfig } from '../../utils/api';
import { CSS_CLASSES } from '../../utils/constants';

const SIZES = ['256x256', '384x384', '512x512', '640x640', '768x768', '1024x1024'];

export default function ModeEditor() {
  const [config, setConfig] = useState(null);       // { default_mode, model_root, lora_root, modes }
  const [inventory, setInventory] = useState({ models: [], loras: [] });
  const [editing, setEditing] = useState(null);      // mode name being edited, or "__new__"
  const [draft, setDraft] = useState(null);           // draft form state
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const apiClientRef = useRef(null);
  
  if (!apiClientRef.current) {
    const config = createApiConfig();
    apiClientRef.current = createApiClient(config);
  }

  const api = apiClientRef.current;

  const load = useCallback(async () => {
    try {
      const [modesRes, modelsRes, lorasRes] = await Promise.all([
        api.fetchGet('/api/modes'),
        api.fetchGet('/api/inventory/models'),
        api.fetchGet('/api/inventory/loras'),
      ]);
      setConfig({
        default_mode: modesRes.default_mode,
        model_root: modelsRes.model_root,
        lora_root: lorasRes.lora_root,
        modes: modesRes.modes,
      });
      
      setInventory({ models: modelsRes.models, loras: lorasRes.loras });
      setError(null);
    } catch (e) {
      console.error("Fetching Nodes aborted: " + e.message);
      setError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const startEdit = (name) => {
    const mode = config.modes[name];
    setEditing(name);
    setDraft({
      name,
      model: mode.model,
      loras: (mode.loras || []).map(l => typeof l === 'string' ? { path: l, strength: 1.0 } : { ...l }),
      default_size: mode.default_size || '512x512',
      default_steps: mode.default_steps || 4,
      default_guidance: mode.default_guidance || 1.0,
    });
  };

  const startNew = () => {
    setEditing('__new__');
    setDraft({
      name: '',
      model: inventory.models[0] || '',
      loras: [],
      default_size: '512x512',
      default_steps: 4,
      default_guidance: 1.0,
    });
  };

  const cancelEdit = () => { setEditing(null); setDraft(null); };

  const saveDraft = async () => {
    if (!draft.name.trim()) { setError('Mode name is required'); return; }
    setSaving(true);
    setError(null);

    const updated = { ...config.modes };
    // If renaming (editing existing with different name), remove old
    if (editing !== '__new__' && editing !== draft.name) {
      delete updated[editing];
    }
    updated[draft.name] = {
      model: draft.model,
      loras: draft.loras,
      default_size: draft.default_size,
      default_steps: draft.default_steps,
      default_guidance: draft.default_guidance,
    };

    const defaultMode = config.default_mode in updated ? config.default_mode : Object.keys(updated)[0];

    try {
      await api.fetchPut('/api/modes', {
        model_root: config.model_root,
        lora_root: config.lora_root,
        default_mode: defaultMode,
        modes: updated,
      });
      setSuccess('Saved');
      setTimeout(() => setSuccess(null), 2000);
      cancelEdit();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const deleteMode = async (name) => {
    if (name === config.default_mode) return;
    if (!confirm(`Delete mode "${name}"?`)) return;

    try {
      await api.fetchDelete(`/api/modes/${encodeURIComponent(name)}`);
      setSuccess('Deleted');
      setTimeout(() => setSuccess(null), 2000);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const setDefaultMode = async (name) => {
    if (name === config.default_mode) return;

    try {
      await api.fetchPut('/api/modes', {
        model_root: config.model_root,
        lora_root: config.lora_root,
        default_mode: name,
        modes: config.modes,
      });
      setSuccess(`"${name}" is now the default mode`);
      setTimeout(() => setSuccess(null), 2000);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  if (!config) {
    return <div className="p-8 text-center text-muted-foreground">Loading configuration...</div>;
  }

  const modeNames = Object.keys(config.modes);
  console.log("NAMES: " + modeNames);

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Mode Configuration</h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-4 w-4 mr-1" /> Reload
          </Button>
          <Button size="sm" onClick={startNew} disabled={editing !== null}>
            <Plus className="h-4 w-4 mr-1" /> Add Mode
          </Button>
        </div>
      </div>

      {error && <div className="rounded-lg bg-red-100 text-red-800 px-4 py-2 text-sm">{error}</div>}
      {success && <div className="rounded-lg bg-green-100 text-green-800 px-4 py-2 text-sm">{success}</div>}

      {/* Mode list */}
      <div className="space-y-3">
        {modeNames.map((name) => {
          const mode = config.modes[name];
          const isDefault = name === config.default_mode;
          const isEditing = editing === name;

          if (isEditing) {
            return <ModeForm key={name} draft={draft} setDraft={setDraft} inventory={inventory}
              onSave={saveDraft} onCancel={cancelEdit} saving={saving} />;
          }

          return (
            <Card key={name} className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{name}</span>
                  {isDefault && <Badge variant="secondary">default</Badge>}
                </div>
                <div className="flex gap-2">
                  {!isDefault && (
                    <Button variant="outline" size="sm" onClick={() => setDefaultMode(name)} disabled={editing !== null} title="Set as default">
                      <Star className="h-4 w-4" />
                    </Button>
                  )}
                  <Button variant="outline" size="sm" onClick={() => startEdit(name)} disabled={editing !== null}>
                    Edit
                  </Button>
                  {!isDefault && (
                    <Button variant="outline" size="sm" onClick={() => deleteMode(name)} disabled={editing !== null}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="mt-2 text-sm text-muted-foreground space-x-4">
                <span>Model: {mode.model}</span>
                <span>Size: {mode.default_size}</span>
                <span>Steps: {mode.default_steps}</span>
                <span>CFG: {mode.default_guidance}</span>
                {mode.loras?.length > 0 && <span>LoRAs: {mode.loras.length}</span>}
              </div>
            </Card>
          );
        })}
      </div>

      {/* New mode form */}
      {editing === '__new__' && (
        <ModeForm draft={draft} setDraft={setDraft} inventory={inventory}
          onSave={saveDraft} onCancel={cancelEdit} saving={saving} isNew />
      )}
    </div>
  );
}


function ModeForm({ draft, setDraft, inventory, onSave, onCancel, saving, isNew }) {
  const patch = (field, value) => setDraft(d => ({ ...d, [field]: value }));

  const addLora = () => {
    if (inventory.loras.length === 0) return;
    setDraft(d => ({ ...d, loras: [...d.loras, { path: inventory.loras[0], strength: 1.0 }] }));
  };

  const removeLora = (idx) => {
    setDraft(d => ({ ...d, loras: d.loras.filter((_, i) => i !== idx) }));
  };

  const updateLora = (idx, field, value) => {
    setDraft(d => ({
      ...d,
      loras: d.loras.map((l, i) => i === idx ? { ...l, [field]: value } : l),
    }));
  };

  return (
    <Card className="p-4 border-2 border-primary space-y-4">

      <h3 className="font-medium">{isNew ? 'New Mode' : `Editing: ${draft.name}`}</h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>Mode Name</Label>
          <Input value={draft.name} onChange={e => patch('name', e.target.value)}
            disabled={!isNew} placeholder="my-mode" />
        </div>
        <div>
          <Label>Model</Label>
          <Select value={draft.model} onValueChange={v => patch('model', v)}>
            <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
              <SelectValue placeholder="Select Model"/>
              </SelectTrigger>
            <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
              {inventory.models.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}
              {/* Include current value if not in inventory */}
              {draft.model && !inventory.models.includes(draft.model) && (
                <SelectItem value={draft.model}>{draft.model} (current)</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label>Default Size</Label>
          <Select value={draft.default_size} onValueChange={v => patch('default_size', v)}>
            <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
              <SelectValue />
              </SelectTrigger>
            <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
              {SIZES.map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Default Steps</Label>
          <Input type="number" min={1} max={100} value={draft.default_steps}
            onChange={e => patch('default_steps', parseInt(e.target.value) || 1)} />
        </div>
        <div>
          <Label>Default Guidance</Label>
          <Input type="number" min={0} max={30} step={0.1} value={draft.default_guidance}
            onChange={e => patch('default_guidance', parseFloat(e.target.value) || 0)} />
        </div>
      </div>

      {/* LoRAs */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <Label>LoRAs</Label>
          <Button variant="outline" size="sm" onClick={addLora} disabled={inventory.loras.length === 0}>
            <Plus className="h-3 w-3 mr-1" /> Add LoRA
          </Button>
        </div>
        {draft.loras.map((lora, idx) => (
          <div key={idx} className="flex items-center gap-2 mb-2">
            <Select value={lora.path} onValueChange={v => updateLora(idx, 'path', v)}>
              <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER} >
                <SelectValue />
                </SelectTrigger>
              <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
                {inventory.loras.map(l => <SelectItem key={l} value={l}>{l}</SelectItem>)}
                {!inventory.loras.includes(lora.path) && (
                  <SelectItem value={lora.path}>{lora.path} (current)</SelectItem>
                )}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2 w-40">
              <span className="text-xs text-muted-foreground w-6">{lora.strength.toFixed(1)}</span>
              <Slider min={0} max={2} step={0.1} value={[lora.strength]}
                onValueChange={([v]) => updateLora(idx, 'strength', v)} />
            </div>
            <Button variant="ghost" size="sm" onClick={() => removeLora(idx)}>
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        ))}
        {draft.loras.length === 0 && (
          <p className="text-xs text-muted-foreground">No LoRAs configured</p>
        )}
      </div>

      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={onSave} disabled={saving}>
          <Save className="h-4 w-4 mr-1" /> {saving ? 'Saving...' : 'Save'}
        </Button>
      </div>
    </Card>
  );
}
