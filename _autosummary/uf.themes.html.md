# uf.themes

Theme and styling support for uf.

Provides built-in themes, dark mode support, and theme customization.

### Functions

| [`create_theme_toggle_button`](#uf.themes.create_theme_toggle_button)()   | Create HTML for theme toggle button.    |
|---------------------------------------------------------------------------------|-----------------------------------------|
| [`generate_theme_toggle_js`](#uf.themes.generate_theme_toggle_js)()     | Generate JavaScript for theme toggling. |
| [`get_theme`](#uf.themes.get_theme)(name)                | Get a built-in theme by name.           |

### Classes

| [`Theme`](#uf.themes.Theme)(name[, colors, fonts, css])               | UI theme configuration.         |
|--------------------------------------------------------------------------------------------------|---------------------------------|
| [`ThemeConfig`](#uf.themes.ThemeConfig)([default_theme, allow_toggle, ...]) | Configuration for theme system. |

### *class* uf.themes.Theme(name, colors=<factory>, fonts=<factory>, css='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

UI theme configuration.

#### name

Theme name

#### colors

Color palette

#### fonts

Font configuration

#### css

Additional custom CSS

#### to_css()

Convert theme to CSS.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  CSS string

### *class* uf.themes.ThemeConfig(default_theme='light', allow_toggle=True, available_themes=None, custom_theme=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for theme system.

### Example

```pycon
>>> config = ThemeConfig(
...     default_theme='dark',
...     allow_toggle=True,
...     available_themes=['light', 'dark', 'ocean']
... )
```

#### get_css()

Get CSS for all available themes.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Combined CSS string

#### get_js()

Get JavaScript for theme functionality.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  JavaScript string

#### get_toggle_button_html()

Get HTML for theme toggle button.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string or empty if toggle not allowed

### uf.themes.create_theme_toggle_button()

Create HTML for theme toggle button.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string for theme toggle button

### uf.themes.generate_theme_toggle_js()

Generate JavaScript for theme toggling.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  JavaScript code for theme switching

### uf.themes.get_theme(name)

Get a built-in theme by name.

* **Parameters:**
  **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Theme name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Theme`](#uf.themes.Theme)]
* **Returns:**
  Theme object or None

### Example

```pycon
>>> theme = get_theme('dark')
>>> css = theme.to_css()
```
