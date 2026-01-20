-- Admin Dashboard Database Schema
-- Run this script to create/update the tables for admin logging

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    ms_user_id VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    email VARCHAR(255),
    first_seen TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(500) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    response_time_ms INTEGER,
    tools_used TEXT[],
    sql_query TEXT,
    sql_results_count INTEGER,
    error TEXT,
    logs JSONB,
    -- Feedback columns
    feedback TEXT,
    feedback_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_ms_user_id ON users(ms_user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_feedback ON messages(feedback) WHERE feedback IS NOT NULL;

-- Migration: Add feedback columns if they don't exist (for existing tables)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'messages' AND column_name = 'feedback') THEN
        ALTER TABLE messages ADD COLUMN feedback TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'messages' AND column_name = 'feedback_at') THEN
        ALTER TABLE messages ADD COLUMN feedback_at TIMESTAMP;
    END IF;
END $$;

-- Learned rules table (auto-extracted from user feedback)
CREATE TABLE IF NOT EXISTS learned_rules (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,              -- German rule (imperative form)
    category VARCHAR(50),                  -- 'output_format', 'data_display', 'behavior'
    keywords TEXT[],                       -- Trigger keywords for matching
    source_question TEXT,                  -- Original user question
    source_feedback TEXT,                  -- Original feedback text
    confidence_score FLOAT DEFAULT 1.0,
    usage_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rules_active ON learned_rules(is_active);
