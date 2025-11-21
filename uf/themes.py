"""Theme and styling support for uf.

Provides built-in themes, dark mode support, and theme customization.
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Theme:
    """UI theme configuration.

    Attributes:
        name: Theme name
        colors: Color palette
        fonts: Font configuration
        css: Additional custom CSS
    """

    name: str
    colors: dict[str, str] = field(default_factory=dict)
    fonts: dict[str, str] = field(default_factory=dict)
    css: str = ""

    def to_css(self) -> str:
        """Convert theme to CSS.

        Returns:
            CSS string
        """
        css_vars = []

        # Add color variables
        for key, value in self.colors.items():
            css_vars.append(f"  --color-{key}: {value};")

        # Add font variables
        for key, value in self.fonts.items():
            css_vars.append(f"  --font-{key}: {value};")

        css = ":root {\n" + "\n".join(css_vars) + "\n}\n\n" + self.css

        return css


# Built-in theme definitions

LIGHT_THEME = Theme(
    name="light",
    colors={
        "primary": "#4CAF50",
        "secondary": "#2196F3",
        "success": "#4CAF50",
        "error": "#f44336",
        "warning": "#ff9800",
        "info": "#2196F3",
        "background": "#ffffff",
        "surface": "#f5f5f5",
        "text": "#212121",
        "text-secondary": "#666666",
        "border": "#dddddd",
    },
    fonts={
        "main": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "mono": "'Courier New', Courier, monospace",
    },
)


DARK_THEME = Theme(
    name="dark",
    colors={
        "primary": "#66BB6A",
        "secondary": "#42A5F5",
        "success": "#66BB6A",
        "error": "#EF5350",
        "warning": "#FFA726",
        "info": "#42A5F5",
        "background": "#1e1e1e",
        "surface": "#2d2d2d",
        "text": "#e0e0e0",
        "text-secondary": "#b0b0b0",
        "border": "#404040",
    },
    fonts={
        "main": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "mono": "'Courier New', Courier, monospace",
    },
    css="""
body {
    background-color: var(--color-background);
    color: var(--color-text);
}

#sidebar {
    background-color: var(--color-surface);
    border-right-color: var(--color-border);
}

#header {
    background-color: var(--color-surface);
    border-bottom-color: var(--color-border);
    color: var(--color-text);
}

.function-item {
    background-color: var(--color-background);
    border-color: var(--color-border);
    color: var(--color-text);
}

.function-item:hover {
    background-color: var(--color-surface);
}

.function-item.active {
    background-color: var(--color-primary);
    color: white;
}

.form-section {
    background-color: var(--color-surface);
    border-color: var(--color-border);
}

#result {
    background-color: var(--color-surface);
    border-color: var(--color-border);
}

#result.success {
    background-color: rgba(102, 187, 106, 0.1);
    border-color: var(--color-success);
}

#result.error {
    background-color: rgba(239, 83, 80, 0.1);
    border-color: var(--color-error);
}

input, textarea, select {
    background-color: var(--color-background);
    border-color: var(--color-border);
    color: var(--color-text);
}

button[type="submit"] {
    background-color: var(--color-primary);
}

button[type="submit"]:hover {
    background-color: var(--color-secondary);
}
""",
)


OCEAN_THEME = Theme(
    name="ocean",
    colors={
        "primary": "#00BCD4",
        "secondary": "#009688",
        "success": "#4CAF50",
        "error": "#f44336",
        "warning": "#FF5722",
        "info": "#03A9F4",
        "background": "#f0f8ff",
        "surface": "#e1f5fe",
        "text": "#01579B",
        "text-secondary": "#0277BD",
        "border": "#B3E5FC",
    },
    fonts={
        "main": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "mono": "'Courier New', Courier, monospace",
    },
    css="""
body {
    background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
}

.function-item.active {
    background: linear-gradient(135deg, #00BCD4 0%, #009688 100%);
}
""",
)


SUNSET_THEME = Theme(
    name="sunset",
    colors={
        "primary": "#FF6F00",
        "secondary": "#F4511E",
        "success": "#558B2F",
        "error": "#C62828",
        "warning": "#F57C00",
        "info": "#1976D2",
        "background": "#FFF3E0",
        "surface": "#FFE0B2",
        "text": "#E65100",
        "text-secondary": "#EF6C00",
        "border": "#FFCC80",
    },
    fonts={
        "main": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "mono": "'Courier New', Courier, monospace",
    },
    css="""
body {
    background: linear-gradient(135deg, #FFE0B2 0%, #FFCC80 100%);
}

.function-item.active {
    background: linear-gradient(135deg, #FF6F00 0%, #F4511E 100%);
}
""",
)


# Theme registry
BUILT_IN_THEMES = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "ocean": OCEAN_THEME,
    "sunset": SUNSET_THEME,
}


def get_theme(name: str) -> Optional[Theme]:
    """Get a built-in theme by name.

    Args:
        name: Theme name

    Returns:
        Theme object or None

    Example:
        >>> theme = get_theme('dark')
        >>> css = theme.to_css()
    """
    return BUILT_IN_THEMES.get(name)


def generate_theme_toggle_js() -> str:
    """Generate JavaScript for theme toggling.

    Returns:
        JavaScript code for theme switching
    """
    js = """
// Theme toggle functionality
(function() {
    const THEME_KEY = 'uf-theme';
    const DEFAULT_THEME = 'light';

    function getTheme() {
        return localStorage.getItem(THEME_KEY) || DEFAULT_THEME;
    }

    function setTheme(theme) {
        localStorage.setItem(THEME_KEY, theme);
        document.body.setAttribute('data-theme', theme);

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('themechange', {
            detail: { theme }
        }));
    }

    function toggleTheme() {
        const current = getTheme();
        const next = current === 'light' ? 'dark' : 'light';
        setTheme(next);
    }

    // Initialize theme on load
    window.addEventListener('load', function() {
        const theme = getTheme();
        document.body.setAttribute('data-theme', theme);
    });

    // Expose functions globally
    window.ufTheme = {
        get: getTheme,
        set: setTheme,
        toggle: toggleTheme
    };
})();
"""
    return js


def create_theme_toggle_button() -> str:
    """Create HTML for theme toggle button.

    Returns:
        HTML string for theme toggle button
    """
    html = """
<button id="theme-toggle" onclick="window.ufTheme.toggle()"
        style="position: fixed; top: 10px; right: 10px; z-index: 1000;
               padding: 10px 15px; border: none; border-radius: 5px;
               cursor: pointer; background: var(--color-surface);
               color: var(--color-text); font-size: 20px;">
    🌓
</button>
<script>
    // Update button on theme change
    window.addEventListener('themechange', function(e) {
        const btn = document.getElementById('theme-toggle');
        btn.textContent = e.detail.theme === 'light' ? '🌙' : '☀️';
    });

    // Set initial icon
    const btn = document.getElementById('theme-toggle');
    btn.textContent = window.ufTheme.get() === 'light' ? '🌙' : '☀️';
</script>
"""
    return html


class ThemeConfig:
    """Configuration for theme system.

    Example:
        >>> config = ThemeConfig(
        ...     default_theme='dark',
        ...     allow_toggle=True,
        ...     available_themes=['light', 'dark', 'ocean']
        ... )
    """

    def __init__(
        self,
        default_theme: str = 'light',
        allow_toggle: bool = True,
        available_themes: Optional[list[str]] = None,
        custom_theme: Optional[Theme] = None,
    ):
        """Initialize theme config.

        Args:
            default_theme: Default theme name
            allow_toggle: Allow users to toggle theme
            available_themes: List of available theme names
            custom_theme: Optional custom theme
        """
        self.default_theme = default_theme
        self.allow_toggle = allow_toggle
        self.available_themes = available_themes or ['light', 'dark']
        self.custom_theme = custom_theme

    def get_css(self) -> str:
        """Get CSS for all available themes.

        Returns:
            Combined CSS string
        """
        css_parts = []

        # Add light theme as default
        light = get_theme('light')
        if light:
            css_parts.append(light.to_css())

        # Add dark theme
        dark = get_theme('dark')
        if dark:
            css_parts.append(f"\nbody[data-theme='dark'] {{\n")
            css_parts.append(dark.to_css())
            css_parts.append("}\n")

        # Add other themes
        for theme_name in self.available_themes:
            if theme_name in ['light', 'dark']:
                continue
            theme = get_theme(theme_name)
            if theme:
                css_parts.append(f"\nbody[data-theme='{theme_name}'] {{\n")
                css_parts.append(theme.to_css())
                css_parts.append("}\n")

        # Add custom theme
        if self.custom_theme:
            css_parts.append(f"\nbody[data-theme='{self.custom_theme.name}'] {{\n")
            css_parts.append(self.custom_theme.to_css())
            css_parts.append("}\n")

        return "\n".join(css_parts)

    def get_js(self) -> str:
        """Get JavaScript for theme functionality.

        Returns:
            JavaScript string
        """
        js = generate_theme_toggle_js()

        # Set default theme
        js += f"\nwindow.ufTheme.set(window.ufTheme.get() || '{self.default_theme}');\n"

        return js

    def get_toggle_button_html(self) -> str:
        """Get HTML for theme toggle button.

        Returns:
            HTML string or empty if toggle not allowed
        """
        if not self.allow_toggle:
            return ""

        return create_theme_toggle_button()
