// index.js
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

if (!process.env.OPENAI_API_KEY) {
  console.warn('⚠️  OPENAI_API_KEY is not set. AI endpoints will fail until you add it.');
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 60 * 1000, // 60s
});

// ============================================
// Middleware
// ============================================

app.use(cors({
  origin: '*', // Change to specific origin(s) in production, e.g. ['https://yourdomain.com']
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// ============================================
// Shared helpers
// ============================================

const MAX_RESPONSES = 20;
const MAX_TEXT_LENGTH = 4000;

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

function cleanText(value, max = MAX_TEXT_LENGTH) {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

/**
 * Calls OpenAI with a strict JSON schema and returns the parsed object.
 */
async function askOpenAIForJson({ system, user, schema, temperature = 0.3, maxTokens }) {
  if (!process.env.OPENAI_API_KEY) {
    throw httpError(500, 'Server is missing OPENAI_API_KEY.');
  }

  let completion;
  try {
    completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature,
      ...(maxTokens ? { max_tokens: maxTokens } : {}),
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      response_format: { type: 'json_schema', json_schema: schema },
    });
  } catch (err) {
    if (err && err.status === 401) {
      throw httpError(500, 'OpenAI authentication failed. Check OPENAI_API_KEY on the server.');
    }
    if (err && err.status === 429) {
      throw httpError(429, 'OpenAI rate limit or quota reached. Please try again shortly.');
    }
    throw httpError(502, 'Could not reach the AI service. Please try again.');
  }

  const raw = completion.choices?.[0]?.message?.content;
  if (!raw) throw httpError(502, 'The AI returned an empty response.');

  try {
    return { data: JSON.parse(raw), model: completion.model };
  } catch (e) {
    throw httpError(502, 'The AI returned an invalid response. Please try again.');
  }
}

// ============================================
// 1) Per-answer check: does the answer make sense?
// ============================================

const EVALUATION_SCHEMA = {
  name: 'answer_evaluation',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    required: ['makes_sense', 'relevance', 'quality', 'comment'],
    properties: {
      makes_sense: {
        type: 'boolean',
        description: 'False if the answer is gibberish, off-topic, contradictory or does not address the question',
      },
      relevance: {
        type: 'string',
        enum: ['on_topic', 'partially_on_topic', 'off_topic', 'unclear'],
      },
      quality: { type: 'number', description: 'Answer quality from 1 to 5' },
      comment: {
        type: 'string',
        description:
          'What the interviewer says out loud next: 1-2 short spoken sentences, max 40 words, plain text, no emojis, no new question',
      },
    },
  },
};

const EVALUATION_PROMPT = `You are Ava, a warm, professional AI interviewer.
You just asked a candidate one interview question and they gave an answer.

First decide whether the answer makes sense: is it coherent, relevant to the question, and meaningful?
- If it is gibberish, off-topic, contradictory, or does not address the question, set makes_sense=false
  and kindly say what was missing or unclear (for example, no example, no result, not relevant).
- If it is on-topic but vague or very short, makes_sense can be true, but say what detail would make it stronger.
- If it is good, acknowledge one specific thing the candidate said.

Rules for "comment":
- It is spoken aloud, so use natural, conversational English in plain text.
- 1-2 sentences, maximum 40 words.
- Refer to something specific from the answer.
- Do NOT ask the next question, do NOT mention scores, and do NOT use emojis or markdown.
- Never invent details the candidate did not say.`;

/**
 * POST /api/evaluate-answer
 * Body: { "category": "Product", "question": "...", "answer": "..." }
 */
app.post('/api/evaluate-answer', async (req, res, next) => {
  try {
    const category = cleanText(req.body?.category, 100) || 'General';
    const question = cleanText(req.body?.question);
    const answer = cleanText(req.body?.answer);

    if (!question || !answer) {
      return res.status(400).json({
        success: false,
        message: 'Both "question" and "answer" are required non-empty strings.',
      });
    }

    const { data, model } = await askOpenAIForJson({
      system: EVALUATION_PROMPT,
      user: `Category: ${category}\n\nQuestion: ${question}\n\nCandidate's answer: ${answer}`,
      schema: EVALUATION_SCHEMA,
      temperature: 0.4,
      maxTokens: 250,
    });

    return res.status(200).json({ success: true, model, evaluation: data });
  } catch (err) {
    next(err);
  }
});

// ============================================
// 2) Full interview analysis (strengths & weaknesses)
// ============================================

function validateInterviewPayload(body) {
  if (!body || typeof body !== 'object') {
    return { error: 'Request body must be a JSON object.' };
  }

  const { responses, candidate } = body;

  if (!Array.isArray(responses) || responses.length === 0) {
    return { error: '"responses" must be a non-empty array of { question, answer } objects.' };
  }
  if (responses.length > MAX_RESPONSES) {
    return { error: `"responses" can contain at most ${MAX_RESPONSES} items.` };
  }

  const cleaned = [];
  for (let i = 0; i < responses.length; i++) {
    const item = responses[i];
    if (!item || typeof item.question !== 'string' || typeof item.answer !== 'string') {
      return { error: `responses[${i}] must have string "question" and "answer" fields.` };
    }
    const question = item.question.trim();
    const answer = item.answer.trim();
    if (!question || !answer) {
      return { error: `responses[${i}] has an empty question or answer.` };
    }
    if (question.length > MAX_TEXT_LENGTH || answer.length > MAX_TEXT_LENGTH) {
      return { error: `responses[${i}] exceeds the ${MAX_TEXT_LENGTH} character limit.` };
    }

    // Optional: the per-answer check the interviewer already did
    const fb = item.feedback && typeof item.feedback === 'object' ? item.feedback : null;

    cleaned.push({
      category: cleanText(item.category, 100) || 'General',
      question,
      answer,
      note: fb && typeof fb.comment === 'string' ? fb.comment.slice(0, 300) : '',
      makesSense: fb && typeof fb.makes_sense === 'boolean' ? fb.makes_sense : null,
    });
  }

  const name = candidate && typeof candidate.name === 'string' ? candidate.name.trim().slice(0, 100) : '';

  return { value: { candidateName: name, responses: cleaned } };
}

const ANALYSIS_SCHEMA = {
  name: 'interview_analysis',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    required: [
      'overall_score',
      'summary',
      'strengths',
      'weaknesses',
      'category_scores',
      'recommended_company_types',
    ],
    properties: {
      overall_score: { type: 'number', description: 'Overall score from 1 to 10' },
      summary: { type: 'string', description: '2-3 sentence overall assessment' },
      strengths: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['area', 'evidence'],
          properties: {
            area: { type: 'string' },
            evidence: { type: 'string', description: 'What in the answers shows this strength' },
          },
        },
      },
      weaknesses: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['area', 'evidence', 'suggestion'],
          properties: {
            area: { type: 'string' },
            evidence: { type: 'string', description: 'What in the answers shows this gap' },
            suggestion: { type: 'string', description: 'A concrete way to improve' },
          },
        },
      },
      category_scores: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['category', 'score', 'comment'],
          properties: {
            category: { type: 'string' },
            score: { type: 'number', description: 'Score from 1 to 5' },
            comment: { type: 'string' },
          },
        },
      },
      recommended_company_types: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['type', 'reason'],
          properties: {
            type: { type: 'string' },
            reason: { type: 'string' },
          },
        },
      },
    },
  },
};

const ANALYSIS_PROMPT = `You are an expert interview coach and hiring consultant.
You will receive a candidate's interview questions and their answers across areas such as
Product, Business, AI Fluency, Technology, Marketing and Sales. The first item may be a short introduction.

Analyze ONLY what is in the answers. Do not invent experience the candidate did not mention.
Be specific, honest and constructive. Reference concrete details from the answers as evidence.
Some answers include an interviewer note; treat answers flagged as not making sense as weak.

Return:
- overall_score (1-10) and a short summary
- 2-5 strengths and 2-5 weaknesses (weaknesses must include an actionable suggestion)
- a score (1-5) and short comment for each scored category that appears in the answers
  (do not score the "Introduction" as a category)
- 2-4 recommended company types the candidate should target, with reasons based on their strongest areas

If an answer is very short or vague, say so and score it accordingly.`;

/**
 * POST /api/analyze
 * Body: {
 *   "candidate": { "name": "Priya" },   // optional
 *   "responses": [ { "category": "Product", "question": "...", "answer": "...", "feedback": {...} } ]
 * }
 */
app.post('/api/analyze', async (req, res, next) => {
  try {
    const { error, value } = validateInterviewPayload(req.body);
    if (error) {
      return res.status(400).json({ success: false, message: error });
    }

    const transcript = value.responses
      .map((r, i) => {
        const flag =
          r.makesSense === false ? '\nInterviewer flag: answer did not make sense or was off-topic' : '';
        const note = r.note ? `\nInterviewer note: ${r.note}` : '';
        return `#${i + 1} [${r.category}]\nQuestion: ${r.question}\nAnswer: ${r.answer}${flag}${note}`;
      })
      .join('\n\n');

    const user =
      (value.candidateName ? `Candidate: ${value.candidateName}\n\n` : '') +
      `Interview transcript:\n\n${transcript}`;

    const { data, model } = await askOpenAIForJson({
      system: ANALYSIS_PROMPT,
      user,
      schema: ANALYSIS_SCHEMA,
      temperature: 0.3,
    });

    return res.status(200).json({ success: true, model, analysis: data });
  } catch (err) {
    next(err);
  }
});

// ============================================
// Basic routes
// ============================================

app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'hi..',
  });
});

app.get('/api/hi', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'hi..',
    timestamp: new Date().toISOString(),
  });
});

app.post('/api/echo', (req, res) => {
  res.status(200).json({
    success: true,
    received: req.body,
  });
});

// ============================================
// 404 Handler
// ============================================
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
  });
});

// ============================================
// Global Error Handler
// ============================================
app.use((err, req, res, next) => {
  console.error('Error:', err.stack || err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal Server Error',
  });
});

// ============================================
// Start Server
// ============================================
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});

module.exports = app;