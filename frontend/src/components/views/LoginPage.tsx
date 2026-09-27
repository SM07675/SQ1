import React, { useState, useEffect } from "react";
import { Eye, EyeOff, Satellite, Zap, Globe, Shield, ArrowRight, Sparkles } from "lucide-react";

// ── Demo Account Credentials ──────────────────────────────────────────────────
export const DEMO_ACCOUNTS = [
  {
    email: "demo@satquery.ai",
    password: "SatQuery2026!",
    name: "Harshit S.",
    role: "Active Analyst",
    initials: "HS",
    organization: "ISRO / SIH 2026",
  },
  {
    email: "analyst@satquery.ai",
    password: "Analyst@123",
    name: "Demo Analyst",
    role: "Remote Sensing Expert",
    initials: "DA",
    organization: "SatQuery Demo",
  },
];

export interface AuthUser {
  email: string;
  name: string;
  role: string;
  initials: string;
  organization: string;
}

interface LoginPageProps {
  theme: "light" | "dark";
  onLogin: (user: AuthUser) => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ theme, onLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [animIn, setAnimIn] = useState(false);
  const [loggingIn, setLoggingIn] = useState(false);
  const [demoExpanded, setDemoExpanded] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimIn(true), 80);
    return () => clearTimeout(t);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    await new Promise((res) => setTimeout(res, 900));

    const match = DEMO_ACCOUNTS.find(
      (a) => a.email.toLowerCase() === email.trim().toLowerCase() && a.password === password
    );

    if (match) {
      // Fade out the login page before transitioning to the app
      setLoggingIn(true);
      await new Promise((res) => setTimeout(res, 350));
      onLogin({
        email: match.email,
        name: match.name,
        role: match.role,
        initials: match.initials,
        organization: match.organization,
      });
    } else {
      setError("Invalid email or password. Try the demo account below.");
      setLoading(false);
    }
  };

  const fillDemo = (account: typeof DEMO_ACCOUNTS[0]) => {
    setEmail(account.email);
    setPassword(account.password);
    setError(null);
    setDemoExpanded(false);
  };

  const features = [
    { icon: <Satellite size={18} />, label: "Satellite Image Analysis" },
    { icon: <Zap size={18} />, label: "AI-Powered Insights" },
    { icon: <Globe size={18} />, label: "Geospatial Intelligence" },
    { icon: <Shield size={18} />, label: "Secure & Verified Data" },
  ];

  return (
    <div className={`login-page-root ${animIn ? "anim-in" : ""} ${loggingIn ? "logging-in" : ""}`} data-theme={theme}>
      {/* Left panel - Branding / Feature Highlights */}
      <div className="login-left-panel">
        <div className="login-brand-block">
          <div className="login-brand-icon">
            <Satellite size={28} strokeWidth={2} />
          </div>
          <span className="login-brand-name">SatQuery</span>
        </div>

        <div className="login-hero-text">
          <h1 className="login-hero-title">
            Analyse the Earth<br />
            <span className="login-hero-gradient">from Space</span>
          </h1>
          <p className="login-hero-sub">
            Ask natural-language questions about satellite imagery. Get AI-powered geospatial intelligence in seconds.
          </p>
        </div>

        <ul className="login-feature-list">
          {features.map((f, i) => (
            <li key={i} className="login-feature-item" style={{ animationDelay: `${0.35 + i * 0.08}s` }}>
              <span className="login-feature-icon">{f.icon}</span>
              <span>{f.label}</span>
            </li>
          ))}
        </ul>

        <div className="login-badge">
          <Sparkles size={13} />
          <span>SIH 2026 - Problem ID 26167</span>
        </div>
      </div>

      {/* Right panel - Login Form */}
      <div className="login-right-panel">
        <div className="login-card">
          {/* Header */}
          <div className="login-card-header">
            <div className="login-card-icon">
              <Satellite size={24} strokeWidth={2.5} />
            </div>
            <h2 className="login-card-title">Welcome back</h2>
            <p className="login-card-sub">Sign in to your SatQuery workspace</p>
          </div>

          {/* Form */}
          <form className="login-form" onSubmit={handleSubmit} noValidate>
            <div className="login-field-group">
              <label className="login-label" htmlFor="login-email">Email address</label>
              <input
                id="login-email"
                type="email"
                autoComplete="email"
                className={`login-input ${error ? "error" : ""}`}
                placeholder="you@satquery.ai"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setError(null); }}
                required
              />
            </div>

            <div className="login-field-group">
              <label className="login-label" htmlFor="login-password">Password</label>
              <div className="login-input-wrap">
                <input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  className={`login-input ${error ? "error" : ""}`}
                  placeholder="**********"
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setError(null); }}
                  required
                />
                <button
                  type="button"
                  className="login-eye-btn"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div className="login-error-banner" role="alert">
                <span>{error}</span>
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              className="login-submit-btn"
              disabled={loading || !email || !password}
            >
              {loading ? (
                <span className="login-spinner" />
              ) : (
                <>
                  <span>Sign In</span>
                  <ArrowRight size={16} strokeWidth={2.5} />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="login-divider">
            <span className="login-divider-line" />
            <span className="login-divider-text">or try a demo account</span>
            <span className="login-divider-line" />
          </div>

          {/* Demo Accounts */}
          <div className="login-demo-section">
            <button
              type="button"
              className="login-demo-toggle"
              onClick={() => setDemoExpanded(!demoExpanded)}
              aria-expanded={demoExpanded}
            >
              <Sparkles size={14} />
              <span>Demo Accounts</span>
              <span className={`login-demo-chevron ${demoExpanded ? "open" : ""}`}>
                {demoExpanded ? "▴" : "▾"}
              </span>
            </button>

            {demoExpanded && (
              <div className="login-demo-list">
                {DEMO_ACCOUNTS.map((account, i) => (
                  <button
                    key={i}
                    type="button"
                    className="login-demo-card"
                    onClick={() => fillDemo(account)}
                  >
                    <div className="login-demo-avatar">
                      <span>{account.initials}</span>
                    </div>
                    <div className="login-demo-info">
                      <span className="login-demo-name">{account.name}</span>
                      <span className="login-demo-role">{account.role}</span>
                      <span className="login-demo-email">{account.email}</span>
                    </div>
                    <span className="login-demo-use">Use</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <p className="login-footer-note">
            Prototype · SIH 2026 · Problem Statement 26167<br />
            <span>All analysis is AI-assisted. Verify critical results.</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
