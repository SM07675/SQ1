import React, { useState, useEffect } from "react";
import { Eye, EyeOff, Satellite, Zap, Globe, Shield, ArrowRight, Sparkles, UserCheck, CheckCircle2 } from "lucide-react";

// ── Demo Account Credentials ──────────────────────────────────────────────────
export const DEMO_ACCOUNTS = [
  {
    email: "demo@satquery.ai",
    password: "SatQuery2026!",
    name: "Harshit S.",
    role: "Lead Analyst",
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
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [animIn, setAnimIn] = useState(false);
  const [loggingIn, setLoggingIn] = useState(false);
  const [demoExpanded, setDemoExpanded] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimIn(true), 80);
    return () => clearTimeout(t);
  }, []);

  const executeLogin = async (user: AuthUser) => {
    setLoggingIn(true);
    await new Promise((res) => setTimeout(res, 350));
    onLogin(user);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    await new Promise((res) => setTimeout(res, 600));

    const trimmedEmail = email.trim().toLowerCase();
    const match = DEMO_ACCOUNTS.find(
      (a) => a.email.toLowerCase() === trimmedEmail && a.password === password
    );

    if (match) {
      await executeLogin({
        email: match.email,
        name: match.name,
        role: match.role,
        initials: match.initials,
        organization: match.organization,
      });
    } else if (trimmedEmail.includes("@") && password.length >= 3) {
      // Universal Access: Support ANY evaluator, judge, or custom user smoothly
      const localPart = trimmedEmail.split("@")[0];
      const formattedName =
        localPart
          .split(/[._-]/)
          .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
          .join(" ") || "Verified Analyst";
      const initials = (formattedName.split(" ").map((w) => w[0]).join("") || "VA").slice(0, 2).toUpperCase();

      await executeLogin({
        email: email.trim(),
        name: formattedName,
        role: "Senior Geospatial Analyst",
        initials: initials,
        organization: "SIH 2026 / ISRO Workspace",
      });
    } else {
      setError("Please enter a valid email address and password (min 3 characters).");
      setLoading(false);
    }
  };

  const handleInstantDemoLogin = async (account: typeof DEMO_ACCOUNTS[0]) => {
    setEmail(account.email);
    setPassword(account.password);
    setError(null);
    setLoading(true);
    await new Promise((res) => setTimeout(res, 350));
    await executeLogin({
      email: account.email,
      name: account.name,
      role: account.role,
      initials: account.initials,
      organization: account.organization,
    });
  };

  const handleGuestLogin = async () => {
    setLoading(true);
    await new Promise((res) => setTimeout(res, 300));
    await executeLogin({
      email: "guest.evaluator@sih.gov.in",
      name: "Guest Evaluator",
      role: "SIH Jury / Evaluator",
      initials: "GE",
      organization: "SIH 2026 Evaluation Panel",
    });
  };

  const fillDemo = (account: typeof DEMO_ACCOUNTS[0]) => {
    setEmail(account.email);
    setPassword(account.password);
    setError(null);
  };

  const features = [
    { icon: <Satellite size={18} />, label: "Satellite Image Analysis (GeoTIFF, Multispectral)" },
    { icon: <Zap size={18} />, label: "AI-Powered Visual & Natural Language Querying" },
    { icon: <Globe size={18} />, label: "Dual-Node E2E Scale-to-Zero Architecture" },
    { icon: <Shield size={18} />, label: "Scientific GDAL Raster Verification & Audit Trail" },
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
          <span>SIH 2026 · Problem Statement 26167 · ISRO</span>
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

          {/* 1-Click Quick Demo Login Cards */}
          <div className="login-quick-cards">
            {DEMO_ACCOUNTS.map((acc, idx) => (
              <button
                key={idx}
                type="button"
                className="login-quick-card"
                onClick={() => handleInstantDemoLogin(acc)}
                title={`Instant 1-Click Sign In as ${acc.name}`}
              >
                <div className="login-quick-avatar">
                  <span>{acc.initials}</span>
                </div>
                <div className="login-quick-info">
                  <span className="login-quick-name">{acc.name}</span>
                  <span className="login-quick-role">{acc.role}</span>
                </div>
                <Sparkles size={13} style={{ color: "#38bdf8", flexShrink: 0 }} />
              </button>
            ))}
          </div>

          {/* Credentials Auto-Fill Hint */}
          <div className="login-hint-pill">
            <span>
              Demo: <span className="login-hint-code">demo@satquery.ai</span> / <span className="login-hint-code">SatQuery2026!</span>
            </span>
            <button
              type="button"
              className="login-hint-action"
              onClick={() => fillDemo(DEMO_ACCOUNTS[0])}
            >
              Fill
            </button>
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
                placeholder="you@satquery.ai or custom email"
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
                  placeholder="Enter your password"
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

            <div className="login-remember-row">
              <label className="login-remember-label">
                <input
                  type="checkbox"
                  className="login-remember-checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                />
                <span>Remember this session</span>
              </label>
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

          {/* Guest / Instant Evaluator Access */}
          <button
            type="button"
            className="login-guest-btn"
            onClick={handleGuestLogin}
            title="Instant access without password"
          >
            <UserCheck size={16} />
            <span>⚡ Continue as Guest / Evaluator</span>
          </button>

          {/* Divider */}
          <div className="login-divider" style={{ marginTop: "16px" }}>
            <span className="login-divider-line" />
            <span className="login-divider-text">more options</span>
            <span className="login-divider-line" />
          </div>

          {/* Demo Accounts List Accordion */}
          <div className="login-demo-section">
            <button
              type="button"
              className="login-demo-toggle"
              onClick={() => setDemoExpanded(!demoExpanded)}
              aria-expanded={demoExpanded}
            >
              <Sparkles size={14} />
              <span>All Pre-configured Demo Accounts</span>
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
                      <span className="login-demo-role">{account.role} · {account.organization}</span>
                      <span className="login-demo-email">{account.email}</span>
                    </div>
                    <span className="login-demo-use">Auto Fill</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <p className="login-footer-note">
            Prototype · SIH 2026 · Problem Statement 26167<br />
            <span>ISRO Geospatial AI Analysis Platform. Verify critical results.</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
