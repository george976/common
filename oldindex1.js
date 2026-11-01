// index.js
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const OpenAI = require('openai');

const app = express();
app.set('trust proxy', 1); // correct client IPs behind Render / proxies (used by the rate limiter)

const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

// Speech settings (all use the SAME OPENAI_API_KEY)
const STT_MODEL = process.env.OPENAI_STT_MODEL || 'gpt-4o-transcribe'; // most accurate. Falls back to 'whisper-1' automatically
const TTS_MODEL = process.env.OPENAI_TTS_MODEL || 'tts-1'; // or 'tts-1-hd' / 'gpt-4o-mini-tts'
const DEFAULT_VOICE = process.env.OPENAI_TTS_VOICE || 'nova';
const TTS_INSTRUCTIONS =
  process.env.OPENAI_TTS_INSTRUCTIONS ||
  'Speak like a warm, professional interviewer: natural pace, friendly and encouraging.';
const VOICES = ['alloy', 'echo', 'fable', 'nova', 'onyx', 'shimmer'];

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

// Audio uploads are kept in memory (short spoken answers only)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 12 * 1024 * 1024, files: 1 }, // 12 MB is plenty for a few minutes of speech
});

/**
 * Tiny in-memory rate limiter (per IP). Your API is public (CORS *), so this
 * stops a stranger from burning through your OpenAI credits.
 */
function rateLimit({ windowMs, max }) {
  const hits = new Map();
  setInterval(() => {
    const now = Date.now();
    for (const [key, v] of hits) if (v.reset < now) hits.delete(key);
  }, windowMs).unref();

  return (req, res, next) => {
    const now = Date.now();
    let h = hits.get(req.ip);
    if (!h || h.reset < now) {
      h = { count: 0, reset: now + windowMs };
      hits.set(req.ip, h);
    }
    h.count += 1;
    if (h.count > max) {
      res.set('Retry-After', String(Math.ceil((h.reset - now) / 1000)));
      return res.status(429).json({ success: false, message: 'Too many requests. Please slow down.' });
    }
    next();
  };
}

const audioLimiter = rateLimit({ windowMs: 60 * 1000, max: 60 });
const aiLimiter = rateLimit({ windowMs: 60 * 1000, max: 40 });

// ============================================
// Shared helpers
// ============================================

const MAX_RESPONSES = 20;
const MAX_TEXT_LENGTH = 4000;
const MAX_SPEAK_LENGTH = 1000;

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

function cleanText(value, max = MAX_TEXT_LENGTH) {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

/** Maps an OpenAI SDK error to a friendly HTTP error. */
function mapOpenAIError(err, what) {
  if (err && err.status === 401) {
    return httpError(500, 'OpenAI authentication failed. Check OPENAI_API_KEY on the server.');
  }
  if (err && err.status === 429) {
    return httpError(429, 'OpenAI rate limit or quota reached. Please try again shortly.');
  }
  if (err && (err.status === 400 || err.status === 404)) {
    console.error(`OpenAI rejected the ${what} request:`, err.message);
    return httpError(502, `OpenAI rejected the ${what} request: ${String(err.message || '').slice(0, 200)}`);
  }
  console.error(`OpenAI ${what} error:`, err && err.message);
  return httpError(502, 'Could not reach the AI service. Please try again.');
}

/**
 * Removes any sentence that ends with a question mark.
 * The interviewer's "comment" must never contain a question, so the candidate
 * is never asked two things at once.
 */
function stripQuestions(text) {
  const sentences = (typeof text === 'string' ? text : '').match(/[^.!?]+[.!?]*/g) || [];
  return sentences
    .filter((s) => !/\?\s*$/.test(s.trim()))
    .join('')
    .trim();
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
    throw mapOpenAIError(err, 'chat');
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
// 0) Speech: Whisper transcription + OpenAI text-to-speech
// ============================================

const MIME_TO_EXT = {
  'audio/webm': 'webm',
  'video/webm': 'webm',
  'audio/mp4': 'mp4',
  'video/mp4': 'mp4',
  'audio/m4a': 'm4a',
  'audio/x-m4a': 'm4a',
  'audio/ogg': 'ogg',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
};

/**
 * POST /api/transcribe   (multipart/form-data)
 *   audio:    the recorded clip (webm / mp4 / ogg / wav / mp3)
 *   language: optional ISO-639-1 code, e.g. "en" (improves accuracy and speed)
 *   prompt:   optional short vocabulary hint (max 200 chars)
 * Returns { success, text, model }
 */
app.post('/api/transcribe', audioLimiter, upload.single('audio'), async (req, res, next) => {
  try {
    if (!process.env.OPENAI_API_KEY) throw httpError(500, 'Server is missing OPENAI_API_KEY.');
    if (!req.file || !req.file.buffer || req.file.size < 500) {
      return res.status(400).json({
        success: false,
        message: 'Send an audio clip in the "audio" form field.',
      });
    }

    const mime = String(req.file.mimetype || '').split(';')[0].toLowerCase();
    const ext = MIME_TO_EXT[mime] || 'webm';
    const language = /^[a-z]{2}$/i.test(String(req.body?.language || '')) ? req.body.language.toLowerCase() : undefined;
    const prompt = cleanText(req.body?.prompt, 200) || undefined;

    const file = await OpenAI.toFile(req.file.buffer, `speech.${ext}`, { type: mime || 'audio/webm' });

    // Try the configured model first; fall back to whisper-1 if the account can't use it
    const models = STT_MODEL === 'whisper-1' ? ['whisper-1'] : [STT_MODEL, 'whisper-1'];
    let result;
    let usedModel = STT_MODEL;
    let lastErr;
    for (const m of models) {
      try {
        result = await openai.audio.transcriptions.create({
          file,
          model: m,
          response_format: 'json',
          temperature: 0,
          ...(language ? { language } : {}),
          ...(prompt ? { prompt } : {}),
        });
        usedModel = m;
        break;
      } catch (err) {
        lastErr = err;
        if (![400, 403, 404].includes(err && err.status)) break; // only fall back on "model not available"-type errors
        console.warn(`Transcription with ${m} failed (${err.status}): ${err.message}`);
      }
    }
    if (!result) throw mapOpenAIError(lastErr, 'transcription');

    const text = (typeof result === 'string' ? result : result?.text || '').trim();
    return res.status(200).json({ success: true, text, model: usedModel });
  } catch (err) {
    next(err);
  }
});

/**
 * POST /api/speak
 * Body: { "text": "...", "voice": "nova" }
 * Returns the spoken audio as audio/mpeg.
 */
app.post('/api/speak', audioLimiter, async (req, res, next) => {
  try {
    if (!process.env.OPENAI_API_KEY) throw httpError(500, 'Server is missing OPENAI_API_KEY.');

    const text = cleanText(req.body?.text, MAX_SPEAK_LENGTH);
    if (!text) {
      return res.status(400).json({ success: false, message: '"text" must be a non-empty string.' });
    }
    const requested = cleanText(req.body?.voice, 30).toLowerCase();
    const voice = VOICES.includes(requested) ? requested : VOICES.includes(DEFAULT_VOICE) ? DEFAULT_VOICE : 'nova';

    let speech;
    try {
      speech = await openai.audio.speech.create({
        model: TTS_MODEL,
        voice,
        input: text,
        response_format: 'mp3',
        // "instructions" is only supported by the gpt-4o-*-tts models
        ...(TTS_MODEL.startsWith('gpt-4o') ? { instructions: TTS_INSTRUCTIONS } : {}),
      });
    } catch (err) {
      throw mapOpenAIError(err, 'speech');
    }

    const audio = Buffer.from(await speech.arrayBuffer());
    res.set({
      'Content-Type': 'audio/mpeg',
      'Content-Length': String(audio.length),
      'Cache-Control': 'no-store',
    });
    return res.status(200).send(audio);
  } catch (err) {
    next(err);
  }
});

// ============================================
// 1) Per-answer check + clarifying questions
// ============================================

const EVALUATION_SCHEMA = {
  name: 'answer_evaluation',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    required: [
      'makes_sense',
      'relevance',
      'quality',
      'proceed',
      'comment',
      'clarifying_question',
      'candidate_name',
    ],
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
      proceed: {
        type: 'boolean',
        description:
          'True if the answer is good enough to move on to the next question. False if the interviewer must ask a clarifying question first.',
      },
      comment: {
        type: 'string',
        description:
          'What the interviewer says out loud: a short spoken reaction to the answer. Plain text, no emojis, no new interview question.',
      },
      clarifying_question: {
        type: 'string',
        description:
          'One short, friendly follow-up question asking for what is missing. Empty string when proceed is true.',
      },
      candidate_name: {
        type: 'string',
        description: 'The candidate\'s first name if they stated it in the answer, otherwise an empty string.',
      },
    },
  },
};

const EVALUATION_PROMPT = `You are Ava, a warm, professional AI interviewer.
You asked a candidate one question and they answered. The answer text may combine several attempts:
if you asked a clarifying question earlier, the candidate's follow-up is appended to the earlier answer.
The answer was usually SPOKEN and transcribed automatically, so ignore small transcription slips, missing
punctuation and filler words. Judge the meaning, not the typing.

Decide whether the candidate gave a proper answer, then set "proceed".

GENERAL RULES
- Be natural and conversational. Your words are spoken aloud: plain text, no emojis, no markdown.
- "comment" must NEVER contain a question and must never end with a question mark. It is a statement only.
  Questions belong only in "clarifying_question", and only when proceed=false.
  When proceed=true, do not ask anything at all (no follow-ups like "What other sources do you use?"),
  because the interviewer will move straight on to the next interview question.
- If a word or phrase makes no sense in context (for example "centipede of a group of people" where "sample" was
  likely meant), it is probably a speech-to-text error, not the candidate's fault. Do not blame them.
  Judge the rest of the answer. If the meaning is still unclear because of such errors, set proceed=false, say
  "I may have misheard that" in the comment, and ask them to repeat or rephrase just the unclear part.
- Never ask the next interview question. Never mention scores. Never invent details the candidate did not say.
- Ask at most ONE clarifying question, and only for what is actually missing.
- If the answer is vague but a genuine attempt, set proceed=true and use "comment" to say what extra detail would make it stronger.

WHEN category is "Introduction"
- A proper introduction includes (1) the candidate's NAME and (2) at least some background or their current role.
  What they are looking for next is nice to have but not required.
- If the NAME is missing, proceed=false and politely ask for their name (and whatever else is missing, in one question).
- If the name is given but there is no background or current role, proceed=false and ask about their background or current role.
- If the intro is complete, proceed=true and greet them warmly by first name in "comment" (max 30 words).
- Always fill "candidate_name" with the first name when it was stated, otherwise "".

FOR ALL OTHER CATEGORIES
- proceed=false (and makes_sense=false where relevant) when the answer is gibberish, a refusal or non-answer
  (e.g. "idk", "ok", "no idea"), clearly off-topic, contradictory, or does not attempt the question.
  Then ask ONE specific clarifying question that helps them answer, such as asking for a concrete example,
  their role in it, or what result it had.
- proceed=true when they genuinely attempt the question, even if imperfectly.
- When proceed=true, "comment" is 1-2 sentences (max 40 words) that reference something specific from the answer.

WHEN proceed=false
- "comment" is a very short, kind reaction (max 15 words) that says what was missing, without scolding.
- "clarifying_question" is the single follow-up question.
- When proceed=true, "clarifying_question" must be an empty string.

ATTEMPT LIMIT
- You are told the current attempt number and the maximum number of clarifications.
  If attempt >= max, you MUST set proceed=true, give a short, kind comment, and leave "clarifying_question" empty.`;

/**
 * POST /api/evaluate-answer
 * Body: {
 *   "category": "Introduction" | "Product" | ...,
 *   "question": "...",
 *   "answer": "...",            // may include earlier attempts appended
 *   "attempt": 0,               // how many clarifications were already asked
 *   "maxAttempts": 2            // optional, default 2
 * }
 */
app.post('/api/evaluate-answer', aiLimiter, async (req, res, next) => {
  try {
    const category = cleanText(req.body?.category, 100) || 'General';
    const question = cleanText(req.body?.question);
    const answer = cleanText(req.body?.answer);

    const attemptRaw = Number(req.body?.attempt);
    const maxRaw = Number(req.body?.maxAttempts);
    const attempt = Number.isFinite(attemptRaw) && attemptRaw >= 0 ? Math.floor(attemptRaw) : 0;
    const maxAttempts = Number.isFinite(maxRaw) && maxRaw >= 0 ? Math.min(Math.floor(maxRaw), 5) : 2;

    if (!question || !answer) {
      return res.status(400).json({
        success: false,
        message: 'Both "question" and "answer" are required non-empty strings.',
      });
    }

    const { data, model } = await askOpenAIForJson({
      system: EVALUATION_PROMPT,
      user:
        `Category: ${category}\n` +
        `Current attempt: ${attempt} (maximum clarifications allowed: ${maxAttempts})\n\n` +
        `Question: ${question}\n\n` +
        `Candidate's answer: ${answer}`,
      schema: EVALUATION_SCHEMA,
      temperature: 0.4,
      maxTokens: 300,
    });

    // The comment is a statement only. Questions belong in "clarifying_question".
    data.comment = stripQuestions(data.comment) || 'Thanks for sharing that.';

    // Safety net: never trap the candidate in an endless loop.
    if (attempt >= maxAttempts) {
      data.proceed = true;
      data.clarifying_question = '';
    }
    // Consistency: proceeding means there is nothing left to clarify.
    if (data.proceed) {
      data.clarifying_question = '';
    } else if (!data.clarifying_question) {
      data.clarifying_question = 'Could you tell me a bit more about that?';
    }

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
Answers may include earlier attempts plus follow-ups made after the interviewer asked a clarifying question.
Answers were usually spoken and transcribed automatically, so do not penalise minor transcription slips,
filler words or missing punctuation.

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
app.post('/api/analyze', aiLimiter, async (req, res, next) => {
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
    speech: { stt: STT_MODEL, tts: TTS_MODEL },
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

  // Upload problems (file too large, unexpected field, ...)
  if (err && err.name === 'MulterError') {
    const tooBig = err.code === 'LIMIT_FILE_SIZE';
    return res.status(tooBig ? 413 : 400).json({
      success: false,
      message: tooBig ? 'Audio clip is too large.' : `Upload error: ${err.message}`,
    });
  }

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