"""HTML template generation for uf web interface.

Provides functions to generate the web UI that lists functions and displays
RJSF forms for interacting with them.
"""

from typing import Optional
from collections.abc import Mapping


def generate_index_html(
    function_specs: Mapping,
    *,
    page_title: str = "Function Interface",
    custom_css: Optional[str] = None,
    rjsf_theme: str = "default",
    base_url: str = "",
) -> str:
    """Generate HTML page with RJSF forms for functions.

    Creates a single-page application with:
    - Function list/navigation sidebar
    - RJSF form for selected function
    - Result display area
    - Uses React and RJSF from CDN (no build step required)

    Args:
        function_specs: Mapping from function names to their specs
        page_title: Title for the HTML page
        custom_css: Optional custom CSS to inject
        rjsf_theme: RJSF theme to use ('default', 'material-ui', 'semantic-ui')
        base_url: Base URL for API endpoints

    Returns:
        Complete HTML string for the web interface
    """
    # Get function list for sidebar
    func_list = []
    for name in function_specs:
        spec = function_specs[name]
        func_list.append({
            'name': name,
            'description': spec.get('description', f"Execute {name}")
        })

    # Generate function list HTML
    func_list_html = "\n".join([
        f'''
        <div class="function-item" onclick="loadFunction('{func['name']}')">
            <div class="function-name">{func['name']}</div>
            <div class="function-desc">{func['description'][:100]}</div>
        </div>
        '''
        for func in func_list
    ])

    custom_styles = custom_css or ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>

    <!-- React and ReactDOM -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

    <!-- RJSF and dependencies -->
    <script src="https://unpkg.com/@rjsf/core@5/dist/react-jsonschema-form.js"></script>
    <script src="https://unpkg.com/@rjsf/validator-ajv8@5/dist/react-jsonschema-form-validator-ajv8.js"></script>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}

        #sidebar {{
            width: 300px;
            background: #f5f5f5;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            padding: 20px;
        }}

        #main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        #header {{
            background: #fff;
            border-bottom: 1px solid #ddd;
            padding: 20px 30px;
        }}

        #header h1 {{
            font-size: 24px;
            margin-bottom: 5px;
        }}

        #header .subtitle {{
            color: #666;
            font-size: 14px;
        }}

        #content {{
            flex: 1;
            overflow-y: auto;
            padding: 30px;
        }}

        .function-item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .function-item:hover {{
            background: #f0f0f0;
            border-color: #4CAF50;
        }}

        .function-item.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}

        .function-name {{
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .function-desc {{
            font-size: 12px;
            color: #666;
            line-height: 1.4;
        }}

        .function-item.active .function-desc {{
            color: rgba(255, 255, 255, 0.9);
        }}

        #form-container {{
            max-width: 800px;
        }}

        .form-section {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .form-section h2 {{
            margin-bottom: 10px;
            font-size: 20px;
        }}

        .form-section .description {{
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }}

        #result {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }}

        #result.success {{
            background: #e8f5e9;
            border-color: #4CAF50;
        }}

        #result.error {{
            background: #ffebee;
            border-color: #f44336;
        }}

        #result h3 {{
            margin-bottom: 10px;
        }}

        #result pre {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }}

        .loading.active {{
            display: block;
        }}

        button[type="submit"] {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
        }}

        button[type="submit"]:hover {{
            background: #45a049;
        }}

        {custom_styles}
    </style>
</head>
<body>
    <div id="sidebar">
        <h2 style="margin-bottom: 15px;">Functions</h2>
        {func_list_html}
    </div>

    <div id="main">
        <div id="header">
            <h1>{page_title}</h1>
            <div class="subtitle">Select a function from the sidebar to get started</div>
        </div>

        <div id="content">
            <div id="form-container"></div>
            <div id="loading" class="loading">Processing...</div>
            <div id="result"></div>
        </div>
    </div>

    <script>
        const {{ useState, useEffect }} = React;
        const Form = JSONSchemaForm.default;
        const validator = JSONSchemaFormValidator.default;

        let currentFunction = null;

        function FunctionForm({{ funcName }}) {{
            const [schema, setSchema] = useState(null);
            const [uiSchema, setUiSchema] = useState({{}});
            const [loading, setLoading] = useState(true);

            useEffect(() => {{
                // Load function spec
                fetch(`{base_url}/api/functions/${{funcName}}/spec`)
                    .then(res => res.json())
                    .then(data => {{
                        setSchema(data.schema);
                        setUiSchema(data.uiSchema || {{}});
                        setLoading(false);
                    }})
                    .catch(err => {{
                        console.error('Error loading spec:', err);
                        setLoading(false);
                    }});
            }}, [funcName]);

            const handleSubmit = ({{ formData }}) => {{
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');

                resultDiv.style.display = 'none';
                loadingDiv.classList.add('active');

                fetch(`{base_url}/${{funcName}}`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(formData)
                }})
                .then(res => {{
                    if (!res.ok) {{
                        return res.json().then(err => {{ throw err; }});
                    }}
                    return res.json();
                }})
                .then(data => {{
                    loadingDiv.classList.remove('active');
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `
                        <h3>Result</h3>
                        <pre>${{JSON.stringify(data, null, 2)}}</pre>
                    `;
                    resultDiv.style.display = 'block';
                }})
                .catch(err => {{
                    loadingDiv.classList.remove('active');
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `
                        <h3>Error</h3>
                        <pre>${{JSON.stringify(err, null, 2)}}</pre>
                    `;
                    resultDiv.style.display = 'block';
                }});
            }};

            if (loading) {{
                return React.createElement('div', {{ className: 'loading active' }}, 'Loading form...');
            }}

            if (!schema) {{
                return React.createElement('div', null, 'Error loading function specification');
            }}

            return React.createElement('div', {{ className: 'form-section' }},
                React.createElement('h2', null, funcName),
                React.createElement('div', {{ className: 'description' }}, schema.description || ''),
                React.createElement(Form, {{
                    schema: schema,
                    uiSchema: uiSchema,
                    validator: validator,
                    onSubmit: handleSubmit
                }})
            );
        }}

        function loadFunction(funcName) {{
            currentFunction = funcName;

            // Update sidebar active state
            document.querySelectorAll('.function-item').forEach(item => {{
                item.classList.remove('active');
            }});
            event.currentTarget.classList.add('active');

            // Clear previous results
            document.getElementById('result').style.display = 'none';

            // Render form
            const container = document.getElementById('form-container');
            const root = ReactDOM.createRoot(container);
            root.render(React.createElement(FunctionForm, {{ funcName }}));
        }}
    </script>
</body>
</html>"""

    return html


def generate_error_page(error_message: str, status_code: int = 500) -> str:
    """Generate a simple error page.

    Args:
        error_message: Error message to display
        status_code: HTTP status code

    Returns:
        HTML string for error page
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error {status_code}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }}
        .error-container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
        }}
        h1 {{
            color: #f44336;
            margin-bottom: 20px;
        }}
        p {{
            color: #666;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <h1>Error {status_code}</h1>
        <p>{error_message}</p>
    </div>
</body>
</html>"""
