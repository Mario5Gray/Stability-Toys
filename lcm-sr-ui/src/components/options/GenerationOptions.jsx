export function GenerationOptions({ 
  effective, 
  onSizeChange, 
  onStepsChange, 
  onCfgChange,
  onPromptChange 
}) {
  return (
    <>
      <SizeSelector value={effective.size} onChange={onSizeChange} />
      <StepsSlider value={effective.steps} onChange={onStepsChange} />
      <CfgSlider value={effective.cfg} onChange={onCfgChange} />
      <PromptTextarea value={effective.prompt} onChange={onPromptChange} />
    </>
  );
}