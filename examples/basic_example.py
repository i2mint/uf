"""Basic example of using uf to create a web UI for functions.

This demonstrates the simplest possible usage - just define functions
and pass them to mk_rjsf_app.

To run this example:
    python examples/basic_example.py

Then open http://localhost:8080 in your browser.
"""

from uf import mk_rjsf_app


def add(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        The sum of x and y
    """
    return x + y


def multiply(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        The product of x and y
    """
    return x * y


def greet(name: str, greeting: str = "Hello") -> str:
    """Generate a greeting message.

    Args:
        name: Name of the person to greet
        greeting: The greeting to use (default: "Hello")

    Returns:
        A friendly greeting message
    """
    return f"{greeting}, {name}!"


def is_even(number: int) -> bool:
    """Check if a number is even.

    Args:
        number: The number to check

    Returns:
        True if the number is even, False otherwise
    """
    return number % 2 == 0


if __name__ == '__main__':
    # Create the web app with all our functions
    app = mk_rjsf_app(
        [add, multiply, greet, is_even],
        page_title="Basic Math & Text Functions"
    )

    # Run the server
    print("Starting server at http://localhost:8080")
    print("Press Ctrl+C to stop")

    # For bottle apps, we can call run() directly
    if hasattr(app, 'run'):
        app.run(host='localhost', port=8080, debug=True)
    else:
        # For FastAPI apps
        print("For FastAPI apps, run:")
        print("  uvicorn examples.basic_example:app --reload")
