import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/select';
import { Badge } from '../ui/badge';
import { Trash2, Plus, Save, RefreshCw, Star, Eye, Upload, X, FileJson } from 'lucide-react';
import { createApiClient, createApiConfig } from '../../utils/api';
import { CSS_CLASSES } from '../../utils/constants';

const SIZES = ['256x256', '384x384', '512x512', '640x640', '768x768', '1024x1024'];
const WORKFLOW_DIR = '/app/workflows'; // Default directory for workflow files

export default function WorkflowEditor() {
  const [config, setConfig] = useState(null);         // { default_workflow, workflows }
  const [editing, setEditing] = useState(null);        // workflow name being edited, or "__new__"
  const [draft, setDraft] = useState(null);            // draft form state
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [viewingJson, setViewingJson] = useState(null); // { name, filepath, json }

  const apiClientRef = useRef(null);
  const fileInputRef = useRef(null);

  if (!apiClientRef.current) {
    const cfg = createApiConfig();
    apiClientRef.current = createApiClient(cfg);
  }

  const api = apiClientRef.current;

  const load = useCallback(async () => {
    try {
      const res = await api.fetchGet('/api/workflows');
      setConfig({
        default_workflow: res.default_workflow,
        workflows: res.workflows,
      });
      setError(null);
    } catch (e) {
      console.error("Fetching workflows failed: " + e.message);
      setError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const startEdit = async (name) => {
    try {
      // Fetch full workflow including JSON content
      const workflow = await api.fetchGet(`/api/workflows/${encodeURIComponent(name)}`);
      const hasFilepath = workflow.workflow?.filepath;

      setEditing(name);
      setDraft({
        name: workflow.name,
        display_name: workflow.display_name || name,
        description: workflow.description || '',
        default_size: workflow.default_size || '512x512',
        default_steps: workflow.default_steps || 20,
        default_cfg: workflow.default_cfg || 7.0,
        tags: (workflow.tags || []).join(', '),
        // File-based workflow support
        filepath: hasFilepath ? workflow.workflow.filepath : '',
        // workflow_content is the actual JSON (loaded from file by backend)
        workflow_content: workflow.workflow_content
          ? JSON.stringify(workflow.workflow_content, null, 2)
          : '{}',
        hasFile: !!hasFilepath,
      });
    } catch (e) {
      setError(e.message);
    }
  };

  const startNew = () => {
    setEditing('__new__');
    setDraft({
      name: '',
      display_name: '',
      description: '',
      default_size: '512x512',
      default_steps: 20,
      default_cfg: 7.0,
      tags: '',
      filepath: '',
      workflow_content: '{}',
      hasFile: true, // New workflows default to file-based
    });
  };

  const cancelEdit = () => { setEditing(null); setDraft(null); };

  const saveDraft = async () => {
    if (!draft.name.trim()) { setError('Workflow name is required'); return; }
    if (!draft.display_name.trim()) { setError('Display name is required'); return; }

    // Validate JSON content
    let workflowContent;
    try {
      workflowContent = JSON.parse(draft.workflow_content);
    } catch (e) {
      setError('Invalid JSON: ' + e.message);
      return;
    }

    // Generate filepath if not set (for new file-based workflows)
    let filepath = draft.filepath.trim();
    if (draft.hasFile && !filepath) {
      const filename = draft.name.toUpperCase().replace(/-/g, '_') + '.json';
      filepath = `${WORKFLOW_DIR}/${filename}`;
    }

    setSaving(true);
    setError(null);

    const payload = {
      display_name: draft.display_name,
      description: draft.description,
      default_size: draft.default_size,
      default_steps: draft.default_steps,
      default_cfg: draft.default_cfg,
      tags: draft.tags.split(',').map(t => t.trim()).filter(Boolean),
      // workflow object with filepath as first field
      workflow: draft.hasFile
        ? { filepath: filepath }
        : workflowContent,
      // Send content separately - backend saves to file
      workflow_content: workflowContent,
    };

    try {
      await api.fetchPost(`/api/workflows/${encodeURIComponent(draft.name)}`, payload);
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

  const deleteWorkflow = async (name) => {
    if (name === config.default_workflow) {
      setError("Cannot delete the default workflow");
      return;
    }
    if (!confirm(`Delete workflow "${name}"?`)) return;

    try {
      await api.fetchDelete(`/api/workflows/${encodeURIComponent(name)}`);
      setSuccess('Deleted');
      setTimeout(() => setSuccess(null), 2000);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const setDefaultWorkflow = async (name) => {
    if (name === config.default_workflow) return;

    try {
      await api.fetchPut('/api/workflows', {
        default_workflow: name,
        workflows: config.workflows,
      });
      setSuccess(`"${name}" is now the default workflow`);
      setTimeout(() => setSuccess(null), 2000);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const viewJson = async (name) => {
    try {
      const workflow = await api.fetchGet(`/api/workflows/${encodeURIComponent(name)}`);
      setViewingJson({
        name,
        filepath: workflow.workflow?.filepath || null,
        json: workflow.workflow_content || workflow.workflow
      });
    } catch (e) {
      setError(e.message);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const json = JSON.parse(evt.target.result);
        setDraft(d => ({
          ...d,
          workflow_content: JSON.stringify(json, null, 2),
          hasFile: true,
        }));
        setSuccess('JSON loaded from file');
        setTimeout(() => setSuccess(null), 2000);
      } catch (err) {
        setError('Invalid JSON file: ' + err.message);
      }
    };
    reader.readAsText(file);
    e.target.value = ''; // Reset for re-upload
  };

  if (!config) {
    return <div className="p-8 text-center text-muted-foreground">Loading workflows...</div>;
  }

  const workflowNames = Object.keys(config.workflows);

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Workflow Configuration</h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-4 w-4 mr-1" /> Reload
          </Button>
          <Button size="sm" onClick={startNew} disabled={editing !== null}>
            <Plus className="h-4 w-4 mr-1" /> Add Workflow
          </Button>
        </div>
      </div>

      {error && <div className="rounded-lg bg-red-100 text-red-800 px-4 py-2 text-sm">{error}</div>}
      {success && <div className="rounded-lg bg-green-100 text-green-800 px-4 py-2 text-sm">{success}</div>}

      {/* Workflow list */}
      <div className="space-y-3">
        {workflowNames.map((name) => {
          const wf = config.workflows[name];
          const isDefault = name === config.default_workflow;
          const isEditing = editing === name;
          const filepath = wf.workflow?.filepath;

          if (isEditing) {
            return (
              <WorkflowForm
                key={name}
                draft={draft}
                setDraft={setDraft}
                onSave={saveDraft}
                onCancel={cancelEdit}
                saving={saving}
                onFileUpload={handleFileUpload}
                fileInputRef={fileInputRef}
              />
            );
          }

          return (
            <Card key={name} className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{wf.display_name || name}</span>
                  {isDefault && <Badge variant="secondary">default</Badge>}
                </div>
                <div className="flex gap-2">
                  {!isDefault && (
                    <Button variant="outline" size="sm" onClick={() => setDefaultWorkflow(name)} disabled={editing !== null} title="Set as default">
                      <Star className="h-4 w-4" />
                    </Button>
                  )}
                  <Button variant="outline" size="sm" onClick={() => viewJson(name)} disabled={editing !== null} title="View JSON">
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => startEdit(name)} disabled={editing !== null}>
                    Edit
                  </Button>
                  {!isDefault && (
                    <Button variant="outline" size="sm" onClick={() => deleteWorkflow(name)} disabled={editing !== null}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              {wf.description && (
                <p className="mt-1 text-sm text-muted-foreground">{wf.description}</p>
              )}
              <div className="mt-2 text-sm text-muted-foreground space-x-4">
                <span>Size: {wf.default_size}</span>
                <span>Steps: {wf.default_steps}</span>
                <span>CFG: {wf.default_cfg}</span>
              </div>
              {filepath && (
                <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground font-mono">
                  <FileJson className="h-3 w-3" />
                  {filepath}
                </div>
              )}
              {wf.tags?.length > 0 && (
                <div className="mt-2 flex gap-1 flex-wrap">
                  {wf.tags.map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                  ))}
                </div>
              )}
            </Card>
          );
        })}
      </div>

      {/* New workflow form */}
      {editing === '__new__' && (
        <WorkflowForm
          draft={draft}
          setDraft={setDraft}
          onSave={saveDraft}
          onCancel={cancelEdit}
          saving={saving}
          isNew
          onFileUpload={handleFileUpload}
          fileInputRef={fileInputRef}
        />
      )}

      {/* JSON Viewer Modal */}
      {viewingJson && (
        <JsonViewerModal
          name={viewingJson.name}
          filepath={viewingJson.filepath}
          json={viewingJson.json}
          onClose={() => setViewingJson(null)}
        />
      )}
    </div>
  );
}


function WorkflowForm({ draft, setDraft, onSave, onCancel, saving, isNew, onFileUpload, fileInputRef }) {
  const patch = (field, value) => setDraft(d => ({ ...d, [field]: value }));

  // Generate suggested filepath from name
  const suggestFilepath = () => {
    if (draft.name) {
      const filename = draft.name.toUpperCase().replace(/-/g, '_') + '.json';
      patch('filepath', `${WORKFLOW_DIR}/${filename}`);
    }
  };

  return (
    <Card className="p-4 border-2 border-primary space-y-4">
      <h3 className="font-medium">{isNew ? 'New Workflow' : `Editing: ${draft.display_name}`}</h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>Name (slug)</Label>
          <Input
            value={draft.name}
            onChange={e => patch('name', e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))}
            disabled={!isNew}
            placeholder="my-workflow"
            onBlur={() => isNew && !draft.filepath && suggestFilepath()}
          />
        </div>
        <div>
          <Label>Display Name</Label>
          <Input
            value={draft.display_name}
            onChange={e => patch('display_name', e.target.value)}
            placeholder="My Custom Workflow"
          />
        </div>
      </div>

      <div>
        <Label>Description</Label>
        <Input
          value={draft.description}
          onChange={e => patch('description', e.target.value)}
          placeholder="Optional description"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label>Default Size</Label>
          <Select value={draft.default_size} onValueChange={v => patch('default_size', v)}>
            <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
              <SelectValue /></SelectTrigger>
            <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
              {SIZES.map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Default Steps</Label>
          <Input
            type="number"
            min={1}
            max={150}
            value={draft.default_steps}
            onChange={e => patch('default_steps', parseInt(e.target.value) || 1)}
          />
        </div>
        <div>
          <Label>Default CFG</Label>
          <Input
            type="number"
            min={0}
            max={30}
            step={0.5}
            value={draft.default_cfg}
            onChange={e => patch('default_cfg', parseFloat(e.target.value) || 0)}
          />
        </div>
      </div>

      <div>
        <Label>Tags (comma-separated)</Label>
        <Input
          value={draft.tags}
          onChange={e => patch('tags', e.target.value)}
          placeholder="txt2img, fast, experimental"
        />
      </div>

      {/* Filepath field */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <Label>Workflow File Path</Label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={draft.hasFile}
              onChange={e => patch('hasFile', e.target.checked)}
              className="rounded"
            />
            Save as file
          </label>
        </div>
        {draft.hasFile ? (
          <div className="flex gap-2">
            <Input
              value={draft.filepath}
              onChange={e => patch('filepath', e.target.value)}
              placeholder="/app/workflows/MY_WORKFLOW.json"
              className="font-mono text-sm"
            />
            <Button variant="outline" size="sm" onClick={suggestFilepath} title="Auto-generate from name">
              Auto
            </Button>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">Workflow will be stored inline in workflows.yml</p>
        )}
      </div>

      {/* Workflow JSON content */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <Label>Workflow JSON {draft.hasFile && <span className="text-muted-foreground font-normal">(will be saved to file)</span>}</Label>
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={onFileUpload}
              className="hidden"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-3 w-3 mr-1" /> Upload JSON
            </Button>
          </div>
        </div>
        <textarea
          className="w-full h-64 p-3 font-mono text-xs border rounded-lg bg-background resize-y"
          value={draft.workflow_content}
          onChange={e => patch('workflow_content', e.target.value)}
          placeholder='{"3": {"class_type": "KSampler", ...}}'
        />
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


function JsonViewerModal({ name, filepath, json, onClose }) {
  const jsonStr = JSON.stringify(json, null, 2);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-background rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h3 className="font-semibold">Workflow JSON: {name}</h3>
            {filepath && (
              <p className="text-xs text-muted-foreground font-mono flex items-center gap-1">
                <FileJson className="h-3 w-3" /> {filepath}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={copyToClipboard}>
              {copied ? 'Copied!' : 'Copy'}
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-4">
          <pre className="font-mono text-xs whitespace-pre-wrap">{jsonStr}</pre>
        </div>
      </div>
    </div>
  );
}
