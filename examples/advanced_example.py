"""Advanced example showing customization options in uf.

This demonstrates:
- Custom page title and CSS
- Using the UfApp class for object-oriented interface
- Functions with more complex types
- Accessing function specs programmatically

To run this example:
    python examples/advanced_example.py
"""

from uf import UfApp


def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate Body Mass Index.

    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters

    Returns:
        Dictionary with BMI value and category
    """
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "bmi": round(bmi, 2),
        "category": category,
    }


def reverse_string(text: str, uppercase: bool = False) -> str:
    """Reverse a string.

    Args:
        text: The string to reverse
        uppercase: Whether to convert to uppercase

    Returns:
        The reversed string
    """
    reversed_text = text[::-1]
    return reversed_text.upper() if uppercase else reversed_text


def fibonacci(n: int) -> list:
    """Generate Fibonacci sequence.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of Fibonacci numbers
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])

    return fib


# Custom CSS for a nicer look
CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

#sidebar {
    background: rgba(255, 255, 255, 0.95);
}

#header {
    background: rgba(255, 255, 255, 0.95);
}

.function-item.active {
    background: #667eea;
}

.function-item:hover {
    border-color: #667eea;
}
"""


if __name__ == '__main__':
    # Create UfApp with customization
    uf_app = UfApp(
        [calculate_bmi, reverse_string, fibonacci],
        page_title="Advanced Function UI",
        custom_css=CUSTOM_CSS,
    )

    # Access function specs programmatically
    print("Available functions:")
    for func_name in uf_app.list_functions():
        spec = uf_app.get_spec(func_name)
        print(f"  - {func_name}: {spec['description']}")

    # Test calling functions directly
    print("\nDirect function calls:")
    print(f"  fibonacci(10) = {uf_app.call('fibonacci', n=10)}")
    print(f"  reverse_string('hello') = {uf_app.call('reverse_string', text='hello')}")

    print("\nStarting server at http://localhost:8080")
    print("Press Ctrl+C to stop")

    # Run the server
    try:
        uf_app.run(host='localhost', port=8080, debug=True)
    except NotImplementedError:
        print("For FastAPI apps, run:")
        print("  uvicorn examples.advanced_example:uf_app.app --reload")
