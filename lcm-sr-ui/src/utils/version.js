export function getFrontendVersion() {
  const version = import.meta.env.VITE_APP_VERSION;
  return version?.trim() || 'dev';
}
