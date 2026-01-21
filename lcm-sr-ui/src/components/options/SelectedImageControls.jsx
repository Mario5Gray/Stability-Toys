export function SelectedImageControls({ 
  selectedParams, 
  onClear, 
  onApplyDelta, 
  onRerun 
}) {
  if (!selectedParams) return null;
  
  return (
    <div className="space-y-2 rounded-2xl border p-3">
      {/* Delta buttons + rerun */}
    </div>
  );
}