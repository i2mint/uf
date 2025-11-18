"""Ultimate showcase of all uf features.

This example demonstrates every major feature of the uf package:
- Async function support
- Pydantic model integration
- Result rendering (tables, charts, images)
- Call history and presets
- Authentication and authorization
- Caching with multiple backends
- Background task execution
- OpenAPI/Swagger documentation
- Webhook integration
- Theme customization
- Field interactions and dependencies
- Custom transformations
- Function grouping and organization

Run with: python examples/ultimate_showcase.py
"""

import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List
import json

# Import Pydantic
try:
    from pydantic import BaseModel, Field, EmailStr
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# Import uf
from uf import (
    # Core
    mk_rjsf_app,
    UfApp,

    # Organization
    FunctionGroup,
    mk_grouped_app,

    # Decorators
    ui_config,
    group,
    field_config,
    with_example,
    requires_auth,
    rate_limit,

    # Field interactions
    with_dependencies,
    FieldDependency,
    DependencyAction,

    # Async support
    is_async_function,
    timeout_async,
    retry_async,

    # Pydantic support
    wrap_pydantic_function,

    # Renderers
    result_renderer,
    register_renderer,
    get_global_renderer_registry,

    # History
    enable_history,
    get_global_history_manager,

    # Authentication
    DictAuthBackend,
    require_auth,
    User,

    # Caching
    cached,
    MemoryCache,
    get_global_cache_backend,

    # Background tasks
    background,
    get_global_task_queue,
    TaskQueue,

    # OpenAPI
    add_openapi_routes,
    OpenAPIConfig,

    # Webhooks
    webhook,
    WebhookManager,
    get_global_webhook_manager,

    # Themes
    ThemeConfig,
    DARK_THEME,
)


# =============================================================================
# 1. PYDANTIC MODELS
# =============================================================================

if HAS_PYDANTIC:
    class UserProfile(BaseModel):
        """User profile with validation."""

        username: str = Field(..., min_length=3, max_length=20, description="Username (3-20 chars)")
        email: EmailStr = Field(..., description="Valid email address")
        age: int = Field(..., gt=0, lt=150, description="Age in years")
        bio: Optional[str] = Field(None, max_length=500, description="Short bio")
        is_active: bool = Field(True, description="Account active?")

    class DataQuery(BaseModel):
        """Query parameters for data analysis."""

        start_date: date = Field(..., description="Start date")
        end_date: date = Field(..., description="End date")
        metric: str = Field(..., description="Metric to analyze")
        granularity: str = Field('daily', description="Time granularity")


# =============================================================================
# 2. AUTHENTICATION SETUP
# =============================================================================

# Create authentication backend
auth_backend = DictAuthBackend.from_dict({
    'admin': {
        'password': 'admin123',
        'roles': ['admin', 'user'],
        'permissions': ['read', 'write', 'delete']
    },
    'user': {
        'password': 'user123',
        'roles': ['user'],
        'permissions': ['read', 'write']
    },
    'viewer': {
        'password': 'view123',
        'roles': ['viewer'],
        'permissions': ['read']
    }
})


# =============================================================================
# 3. CACHE SETUP
# =============================================================================

# Create memory cache with 100 item limit
cache = MemoryCache(max_size=100)


# =============================================================================
# 4. BACKGROUND TASK QUEUE
# =============================================================================

# Create task queue with 2 workers
task_queue = TaskQueue(num_workers=2)
task_queue.start()


# =============================================================================
# 5. WEBHOOK MANAGER
# =============================================================================

webhook_manager = WebhookManager()
# In production, you would add webhook URLs:
# webhook_manager.add_webhook('https://example.com/webhook', events=['success'])


# =============================================================================
# 6. BASIC FUNCTIONS (Group: Utilities)
# =============================================================================

@group('Utilities')
@ui_config(
    title='Add Numbers',
    description='Add two numbers together',
)
@field_config('x', title='First Number', description='Enter the first number')
@field_config('y', title='Second Number', description='Enter the second number')
@with_example(x=10, y=20, example_name='Ten plus twenty')
@cached(ttl=300)  # Cache for 5 minutes
@result_renderer('json')
def add(x: int, y: int) -> dict:
    """Add two numbers and return detailed result."""
    result = x + y
    return {
        'operation': 'addition',
        'operands': [x, y],
        'result': result,
        'is_even': result % 2 == 0,
        'timestamp': datetime.now().isoformat()
    }


@group('Utilities')
@ui_config(title='Calculate Statistics')
@result_renderer('table')
def calculate_stats(numbers: str) -> list[dict]:
    """Calculate statistics from a comma-separated list of numbers.

    Returns a table of statistical measures.
    """
    nums = [float(n.strip()) for n in numbers.split(',')]

    return [
        {'metric': 'Count', 'value': len(nums)},
        {'metric': 'Sum', 'value': sum(nums)},
        {'metric': 'Mean', 'value': sum(nums) / len(nums)},
        {'metric': 'Min', 'value': min(nums)},
        {'metric': 'Max', 'value': max(nums)},
        {'metric': 'Range', 'value': max(nums) - min(nums)},
    ]


# =============================================================================
# 7. ASYNC FUNCTIONS (Group: Async Operations)
# =============================================================================

@group('Async Operations')
@ui_config(title='Async Data Fetch')
@timeout_async(5.0)  # 5 second timeout
@retry_async(max_retries=3, delay=1.0)
@cached(ttl=60)
async def fetch_data(endpoint: str, timeout: float = 2.0) -> dict:
    """Fetch data from an API endpoint (simulated).

    Demonstrates async support with timeout and retry.
    """
    # Simulate async API call
    await asyncio.sleep(timeout)

    return {
        'endpoint': endpoint,
        'status': 'success',
        'data': {
            'items': ['item1', 'item2', 'item3'],
            'count': 3,
            'fetched_at': datetime.now().isoformat()
        },
        'latency_ms': timeout * 1000
    }


@group('Async Operations')
@ui_config(title='Async Batch Processing')
async def process_batch(item_count: int = 5, delay_per_item: float = 0.5) -> dict:
    """Process multiple items concurrently.

    Demonstrates concurrent async execution.
    """
    async def process_item(item_id: int):
        await asyncio.sleep(delay_per_item)
        return f"Item {item_id} processed"

    # Process items concurrently
    results = await asyncio.gather(*[process_item(i) for i in range(item_count)])

    return {
        'total_items': item_count,
        'results': results,
        'total_time_seconds': delay_per_item,  # Concurrent, not sequential
        'completed_at': datetime.now().isoformat()
    }


# =============================================================================
# 8. PYDANTIC FUNCTIONS (Group: Data Management)
# =============================================================================

if HAS_PYDANTIC:
    @group('Data Management')
    @ui_config(title='Create User Profile')
    @result_renderer('json')
    def create_user(profile: UserProfile) -> dict:
        """Create a user profile with full validation.

        Demonstrates Pydantic integration with automatic form generation
        and validation.
        """
        return {
            'status': 'created',
            'profile': profile.dict(),
            'validation': 'All fields validated successfully',
            'created_at': datetime.now().isoformat()
        }

    # Wrap to handle Pydantic models
    create_user = wrap_pydantic_function(create_user)


    @group('Data Management')
    @ui_config(title='Analyze Data Range')
    @result_renderer('table')
    def analyze_data_range(query: DataQuery) -> list[dict]:
        """Analyze data for a specific date range.

        Demonstrates Pydantic models with date fields.
        """
        days = (query.end_date - query.start_date).days + 1

        return [
            {'field': 'Metric', 'value': query.metric},
            {'field': 'Start Date', 'value': query.start_date.isoformat()},
            {'field': 'End Date', 'value': query.end_date.isoformat()},
            {'field': 'Days', 'value': days},
            {'field': 'Granularity', 'value': query.granularity},
            {'field': 'Data Points', 'value': days if query.granularity == 'daily' else days // 7},
        ]

    # Wrap to handle Pydantic models
    analyze_data_range = wrap_pydantic_function(analyze_data_range)


# =============================================================================
# 9. AUTHENTICATED FUNCTIONS (Group: Admin)
# =============================================================================

@group('Admin')
@ui_config(title='View System Status')
@require_auth(auth_backend, roles=['admin', 'user'])
@result_renderer('table')
def get_system_status() -> list[dict]:
    """View system status (requires authentication).

    Accessible by: admin, user
    """
    return [
        {'component': 'Web Server', 'status': 'Running', 'uptime_hours': 48},
        {'component': 'Database', 'status': 'Running', 'uptime_hours': 240},
        {'component': 'Cache', 'status': 'Running', 'uptime_hours': 48},
        {'component': 'Task Queue', 'status': 'Running', 'uptime_hours': 48},
    ]


@group('Admin')
@ui_config(title='Delete Old Data')
@require_auth(auth_backend, roles=['admin'], permissions=['delete'])
@with_example(days_old=30, example_name='Delete 30-day old data')
def delete_old_data(days_old: int = 30, confirm: bool = False) -> dict:
    """Delete data older than specified days (admin only).

    Requires admin role and delete permission.
    """
    if not confirm:
        return {
            'status': 'cancelled',
            'message': 'Confirmation required to delete data',
            'would_delete': f'Data older than {days_old} days'
        }

    return {
        'status': 'deleted',
        'days_old': days_old,
        'deleted_count': 150,  # Simulated
        'deleted_at': datetime.now().isoformat()
    }


# =============================================================================
# 10. BACKGROUND TASKS (Group: Background Jobs)
# =============================================================================

@group('Background Jobs')
@ui_config(title='Send Bulk Emails')
@background(task_queue=task_queue)
def send_bulk_emails(recipient_count: int, delay_per_email: float = 1.0) -> dict:
    """Send emails in the background (returns immediately).

    This function runs in a background worker thread.
    Returns a task_id immediately.
    """
    import time

    results = []
    for i in range(recipient_count):
        time.sleep(delay_per_email)
        results.append(f"Email {i+1} sent to recipient_{i+1}@example.com")

    return {
        'total_sent': recipient_count,
        'results': results,
        'completed_at': datetime.now().isoformat()
    }


@group('Background Jobs')
@ui_config(title='Generate Report')
@background(task_queue=task_queue)
@webhook(on=['success', 'failure'], manager=webhook_manager)
def generate_large_report(pages: int = 100, delay_per_page: float = 0.1) -> dict:
    """Generate a large report in the background.

    Demonstrates background tasks + webhooks.
    Webhook fires on completion or failure.
    """
    import time

    for i in range(pages):
        time.sleep(delay_per_page)

    return {
        'status': 'completed',
        'pages': pages,
        'file_size_mb': pages * 0.5,  # Simulated
        'generated_at': datetime.now().isoformat()
    }


# =============================================================================
# 11. CACHED EXPENSIVE OPERATIONS (Group: Analytics)
# =============================================================================

@group('Analytics')
@ui_config(title='Calculate Prime Numbers')
@cached(ttl=600, backend=cache)  # Cache for 10 minutes
@result_renderer('json')
def calculate_primes(limit: int = 1000) -> dict:
    """Calculate prime numbers up to limit (cached).

    Expensive operation - results are cached for 10 minutes.
    """
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [n for n in range(2, limit + 1) if is_prime(n)]

    return {
        'limit': limit,
        'count': len(primes),
        'primes': primes[:20],  # First 20
        'largest': primes[-1] if primes else None,
        'calculated_at': datetime.now().isoformat(),
        'cached': True
    }


# =============================================================================
# 12. FIELD DEPENDENCIES (Group: Forms)
# =============================================================================

@group('Forms')
@ui_config(title='Conditional Shipping Form')
@with_dependencies(
    FieldDependency(
        source_field='needs_shipping',
        target_field='address',
        action=DependencyAction.SHOW,
        condition=lambda value: value == True
    ),
    FieldDependency(
        source_field='needs_shipping',
        target_field='express_delivery',
        action=DependencyAction.SHOW,
        condition=lambda value: value == True
    )
)
def process_order(
    product: str,
    quantity: int,
    needs_shipping: bool = False,
    address: str = '',
    express_delivery: bool = False
) -> dict:
    """Process an order with conditional shipping fields.

    Demonstrates field dependencies - shipping fields only show
    if needs_shipping is True.
    """
    result = {
        'order_id': f'ORD-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        'product': product,
        'quantity': quantity,
        'needs_shipping': needs_shipping,
        'total_price': quantity * 29.99  # Simulated
    }

    if needs_shipping:
        result['shipping'] = {
            'address': address,
            'express': express_delivery,
            'estimated_days': 1 if express_delivery else 5
        }

    return result


# =============================================================================
# 13. RATE LIMITED FUNCTIONS (Group: API)
# =============================================================================

@group('API')
@ui_config(title='API Endpoint')
@rate_limit(calls=5, period=60)  # 5 calls per minute
def api_call(endpoint: str, method: str = 'GET') -> dict:
    """Make an API call (rate limited to 5/minute).

    Demonstrates rate limiting.
    """
    return {
        'endpoint': endpoint,
        'method': method,
        'status': 200,
        'rate_limit': {
            'limit': 5,
            'period_seconds': 60,
            'remaining': 4  # Simulated
        },
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# 14. HISTORY-ENABLED FUNCTIONS (Group: History)
# =============================================================================

@group('History')
@ui_config(title='Search (with history)')
@enable_history(max_calls=50)
def search(query: str, filters: str = '', limit: int = 10) -> dict:
    """Search with automatic history tracking.

    All calls are recorded in history. You can view past searches
    and reuse parameters as presets.
    """
    return {
        'query': query,
        'filters': filters,
        'limit': limit,
        'results_count': 42,  # Simulated
        'search_time_ms': 23,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# 15. CUSTOM RENDERER EXAMPLE (Group: Visualization)
# =============================================================================

@group('Visualization')
@ui_config(title='Generate Chart Data')
@result_renderer('chart')
def generate_chart_data(data_points: int = 10, chart_type: str = 'line') -> dict:
    """Generate data for visualization.

    Returns data in a format suitable for charting libraries.
    """
    import random

    labels = [f'Point {i+1}' for i in range(data_points)]
    values = [random.randint(10, 100) for _ in range(data_points)]

    return {
        'type': chart_type,
        'labels': labels,
        'datasets': [
            {
                'label': 'Sample Data',
                'data': values,
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'borderColor': 'rgba(75, 192, 192, 1)',
                'borderWidth': 1
            }
        ]
    }


# =============================================================================
# CREATE THE APP
# =============================================================================

# Collect all functions
functions = [
    # Utilities
    add,
    calculate_stats,

    # Async
    fetch_data,
    process_batch,

    # Admin
    get_system_status,
    delete_old_data,

    # Background
    send_bulk_emails,
    generate_large_report,

    # Analytics
    calculate_primes,

    # Forms
    process_order,

    # API
    api_call,

    # History
    search,

    # Visualization
    generate_chart_data,
]

# Add Pydantic functions if available
if HAS_PYDANTIC:
    functions.extend([
        create_user,
        analyze_data_range,
    ])

# Create grouped app with dark theme
app = mk_grouped_app(
    functions,
    page_title='uf Ultimate Showcase',
    theme_config=ThemeConfig(
        default_theme='dark',
        allow_toggle=True,
        available_themes=['light', 'dark', 'ocean', 'sunset']
    ),
    custom_css="""
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .app-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .feature-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 12px;
        font-size: 0.9rem;
    }
    """
)

# Add OpenAPI documentation
openapi_config = OpenAPIConfig(
    title='uf Ultimate Showcase API',
    version='1.0.0',
    description='Comprehensive demonstration of all uf features',
    enable_swagger=True,
    enable_redoc=True
)

add_openapi_routes(app.app, functions, **openapi_config.to_dict())

# Print startup information
print("=" * 70)
print("uf Ultimate Showcase - All Features Demonstrated")
print("=" * 70)
print("\nFeatures included:")
print("  ✓ Async function support (timeout, retry)")
if HAS_PYDANTIC:
    print("  ✓ Pydantic model integration (auto forms + validation)")
else:
    print("  ⚠ Pydantic not installed (install with: pip install pydantic)")
print("  ✓ Result rendering (JSON, tables, charts)")
print("  ✓ Call history and presets")
print("  ✓ Authentication (3 test users)")
print("  ✓ Caching (memory backend)")
print("  ✓ Background tasks (2 worker threads)")
print("  ✓ OpenAPI/Swagger documentation")
print("  ✓ Webhook integration")
print("  ✓ Theme system (dark mode + 4 themes)")
print("  ✓ Field dependencies")
print("  ✓ Rate limiting")
print("  ✓ Function grouping")
print("\nTest Users:")
print("  • admin / admin123 (full access)")
print("  • user / user123 (read + write)")
print("  • viewer / view123 (read only)")
print("\nDocumentation:")
print("  • Swagger UI: http://localhost:8080/docs")
print("  • ReDoc: http://localhost:8080/redoc")
print("  • OpenAPI Spec: http://localhost:8080/openapi.json")
print("\nStarting server on http://localhost:8080")
print("=" * 70)
print()

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
