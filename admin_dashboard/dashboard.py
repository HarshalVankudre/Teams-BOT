"""
Admin Dashboard for Teams Bot
View conversations, users, and feedback.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

# Import admin logger - try direct import first (Docker), then from rag package (local)
try:
    from admin_logger import AdminLogger, AdminConfig
except ImportError:
    from rag.admin_logger import AdminLogger, AdminConfig

# Create admin logger instance
config = AdminConfig.from_env()
admin_logger = AdminLogger(config)

app = Flask(__name__)


@app.route('/health')
def health():
    """Health check endpoint for App Runner."""
    # Get counts if database available
    stats = {}
    if admin_logger.available:
        try:
            stats = admin_logger.get_statistics()
        except Exception as e:
            stats = {'error': str(e)}

    return {
        'status': 'healthy',
        'database': admin_logger.available,
        'config': {
            'host': config.host[:20] + '...' if config.host else 'NOT SET',
            'db': config.database or 'NOT SET',
            'user': config.user or 'NOT SET',
            'password_set': bool(config.password),
        },
        'counts': {
            'users': stats.get('total_users', 0),
            'conversations': stats.get('total_conversations', 0),
            'messages': stats.get('total_messages', 0),
        }
    }, 200


@app.route('/')
def index():
    """Dashboard overview page."""
    stats = admin_logger.get_statistics()
    return render_template('index.html', stats=stats)


@app.route('/conversations')
def conversations():
    """List all conversations with advanced filtering."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    user_filter = request.args.get('user_id', '', type=str)
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    feedback_filter = request.args.get('feedback', '')  # 'with', 'without', or ''
    per_page = 20
    offset = (page - 1) * per_page

    # Convert feedback filter to boolean
    has_feedback = None
    if feedback_filter == 'with':
        has_feedback = True
    elif feedback_filter == 'without':
        has_feedback = False

    # Use advanced filtering if any filter is active
    if search or user_filter or date_from or date_to or feedback_filter:
        convs = admin_logger.get_conversations_filtered(
            user_id=int(user_filter) if user_filter else None,
            date_from=date_from if date_from else None,
            date_to=date_to if date_to else None,
            has_feedback=has_feedback,
            search=search if search else None,
            limit=per_page,
            offset=offset
        )
        total = len(convs) if len(convs) < per_page else per_page * 10  # Estimate
    else:
        convs = admin_logger.get_all_conversations(limit=per_page, offset=offset)
        stats = admin_logger.get_statistics()
        total = stats.get('total_conversations', 0)

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    # Get all users for filter dropdown
    all_users = admin_logger.get_all_users_simple()

    return render_template('conversations.html',
                           conversations=convs,
                           page=page,
                           total_pages=total_pages,
                           search=search,
                           user_filter=user_filter,
                           date_from=date_from,
                           date_to=date_to,
                           feedback_filter=feedback_filter,
                           all_users=all_users)


@app.route('/conversations/<int:conversation_id>')
def conversation_detail(conversation_id):
    """View a single conversation with all messages."""
    conv = admin_logger.get_conversation_with_user(conversation_id)
    if not conv:
        return redirect(url_for('conversations'))

    messages = admin_logger.get_conversation_messages(conversation_id)
    return render_template('conversation.html', conversation=conv, messages=messages)


@app.route('/users')
def users():
    """List all users with search."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    per_page = 20
    offset = (page - 1) * per_page

    if search:
        user_list = admin_logger.search_users(search, limit=per_page)
        total = len(user_list)
    else:
        user_list = admin_logger.get_all_users(limit=per_page, offset=offset)
        stats = admin_logger.get_statistics()
        total = stats.get('total_users', 0)

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return render_template('users.html',
                           users=user_list,
                           page=page,
                           total_pages=total_pages,
                           search=search)


@app.route('/users/<int:user_id>/conversations')
def user_conversations(user_id):
    """View conversations for a specific user."""
    convs = admin_logger.get_user_conversations(user_id)
    return render_template('conversations.html',
                           conversations=convs,
                           page=1,
                           total_pages=1,
                           search='',
                           user_filter=user_id)


@app.route('/feedback')
def feedback():
    """List all feedback entries with search."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    per_page = 20
    offset = (page - 1) * per_page

    if search:
        feedback_list = admin_logger.search_feedback(search, limit=per_page)
        total = len(feedback_list)
    else:
        feedback_list = admin_logger.get_all_feedback(limit=per_page, offset=offset)
        stats = admin_logger.get_statistics()
        total = stats.get('total_feedback', 0)

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return render_template('feedback.html',
                           feedback_list=feedback_list,
                           page=page,
                           total_pages=total_pages,
                           search=search)


@app.route('/rules')
def rules():
    """List all learned rules with management options."""
    include_inactive = request.args.get('show_inactive', '').lower() == 'true'
    rules_list = admin_logger.get_all_rules(include_inactive=include_inactive)

    # Count active rules
    active_count = sum(1 for r in rules_list if r.get('is_active', True))
    total_count = len(rules_list)

    return render_template('rules.html',
                           rules=rules_list,
                           active_count=active_count,
                           total_count=total_count,
                           show_inactive=include_inactive)


# ========== DELETE API ROUTES ==========

@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a single conversation."""
    success = admin_logger.delete_conversation(conversation_id)
    if success:
        return jsonify({'success': True, 'message': 'Gespräch gelöscht'})
    return jsonify({'success': False, 'message': 'Fehler beim Löschen'}), 400


@app.route('/api/users/<int:user_id>/conversations', methods=['DELETE'])
def delete_user_conversations(user_id):
    """Delete all conversations for a user."""
    count = admin_logger.delete_user_conversations(user_id)
    return jsonify({'success': True, 'deleted_count': count, 'message': f'{count} Gespräche gelöscht'})


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user and all their data."""
    success = admin_logger.delete_user(user_id)
    if success:
        return jsonify({'success': True, 'message': 'Benutzer gelöscht'})
    return jsonify({'success': False, 'message': 'Fehler beim Löschen'}), 400


# ========== RULES API ROUTES ==========

@app.route('/api/rules/<int:rule_id>', methods=['DELETE'])
def delete_rule(rule_id):
    """Delete a learned rule."""
    success = admin_logger.delete_rule(rule_id)
    if success:
        return jsonify({'success': True, 'message': 'Regel gelöscht'})
    return jsonify({'success': False, 'message': 'Fehler beim Löschen'}), 400


@app.route('/api/rules/<int:rule_id>/toggle', methods=['POST'])
def toggle_rule(rule_id):
    """Toggle a rule's active status."""
    data = request.get_json() or {}
    is_active = data.get('is_active', True)
    success = admin_logger.toggle_rule_active(rule_id, is_active)
    if success:
        status = 'aktiviert' if is_active else 'deaktiviert'
        return jsonify({'success': True, 'message': f'Regel {status}'})
    return jsonify({'success': False, 'message': 'Fehler beim Ändern'}), 400


@app.template_filter('truncate_text')
def truncate_text(text, length=100):
    """Truncate text to specified length."""
    if not text:
        return ''
    if len(text) <= length:
        return text
    return text[:length] + '...'


@app.template_filter('format_datetime')
def format_datetime(dt):
    """Format datetime for display."""
    if not dt:
        return '-'
    return dt.strftime('%d.%m.%Y %H:%M')


@app.template_filter('format_ms')
def format_ms(ms):
    """Format milliseconds to readable time."""
    if not ms:
        return '-'
    if ms < 1000:
        return f'{ms}ms'
    return f'{ms/1000:.1f}s'


if __name__ == '__main__':
    print("=" * 50)
    print("Admin Dashboard starting...")
    print(f"Database: {config.database}")
    print(f"Available: {admin_logger.available}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
