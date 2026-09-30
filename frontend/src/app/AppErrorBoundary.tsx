import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryState {
  failed: boolean;
}

export class AppErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Diagnostic UI render failed", error, info.componentStack);
  }

  render() {
    if (this.state.failed) {
      return (
        <main className="fatal-error" role="alert">
          <span className="eyebrow">Diagnostic UI error</span>
          <h1>This view could not be rendered.</h1>
          <p>The orchestration service is unaffected. Reload to hydrate the diagnostic state again.</p>
          <button type="button" onClick={() => window.location.reload()}>
            Reload diagnostics
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}
