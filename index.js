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
    console.log(`Transcribed ${req.file.size} bytes with ${usedModel}: "${String(result.text || '').slice(0, 80)}"`);

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
          'What the interviewer says out loud, spoken directly to the candidate using "I" and "you": a short reaction to the answer. Plain text, no emojis, no new interview question. Never refer to the candidate in the third person.',
      },
      clarifying_question: {
        type: 'string',
        description:
          'One short, friendly follow-up question spoken directly to the candidate ("you"/"your") asking for what is missing. Empty string when proceed is true.',
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

HOW YOU SPEAK (very important)
- You are talking TO the candidate, face to face, in a live conversation. You are "I", and the candidate is "you".
  Always use "you" and "your" ("Thanks for walking me through that, I like how you tested the idea with real users.").
- NEVER talk about the candidate in the third person. Never say "the candidate", "the interviewee", "the applicant",
  "he", "she", "they", "them" or "their" about the person you are interviewing, and never use their name as the subject of a
  sentence about them (wrong: "Sam did a great job explaining the roadmap"; right: "You did a great job explaining the roadmap").
  Their first name may only be used as a direct greeting or address ("Nice to meet you, Sam.").
- This applies to "comment" AND "clarifying_question". The notes below are written about "the candidate" only to instruct you;
  do not copy that wording into what you say.

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
- For area questions, a vague or thin answer gets a follow-up question (see below) until the attempt limit is reached.

WHEN category is "Introduction"
- A proper introduction includes (1) the candidate's NAME and (2) at least some background or their current role.
  What they are looking for next is nice to have but not required.
- If the NAME is missing, proceed=false and politely ask for their name (and whatever else is missing, in one question).
- If the name is given but there is no background or current role, proceed=false and ask about their background or current role.
- If the intro is complete, proceed=true and greet them warmly by first name in "comment" (max 30 words), speaking to them as "you".
- Always fill "candidate_name" with the first name when it was stated, otherwise "".

FOR ALL OTHER CATEGORIES (area questions)
You are gathering enough evidence to judge how well this person actually works in the area. One answer is rarely enough.
- You receive the candidate's background, the conversation so far in this area, and the current attempt number
  (the number of follow-ups already asked).
- If you cannot yet judge their practical ability, set proceed=false and ask ONE specific follow-up in "clarifying_question".
  At attempt 0, ask a follow-up unless the answer is already exceptionally concrete (a real example, their own role, a decision, a result).
- Good follow-ups dig into: a concrete example, what THEY did versus the team, a decision and its trade-off, how they measured the result,
  what they would do differently, or how they would handle a changed situation. Tie the follow-up to their background and to
  something specific they just said. Never repeat a question that was already asked.
  Phrase these directly to the person ("What did you personally do versus your team?").
- Judge practical skill, not vocabulary. If the candidate does not know or does not use a technical term (for example RAG) but describes
  sound practice, give them credit. Never quiz them on jargon; ask how they would do it in practice instead.
- Set proceed=true once you have a reasonably clear picture of their depth, or when the attempt limit is reached.
- If the answer is gibberish, a non-answer (e.g. "idk"), off-topic or contradictory, set proceed=false (makes_sense=false where relevant) and ask
  ONE helpful question, such as for a concrete example, their role in it, or what result it had.
- When proceed=true, "comment" is 1-2 sentences (max 40 words) that reference something specific from what they said, spoken to them as "you".

WHEN proceed=false
- "comment" is a very short, kind reaction (max 15 words) that says what was missing, without scolding
  (for example "I didn't catch a specific example there." Not: "The candidate gave no example.").
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
    const background = cleanText(req.body?.background, 1500);
    const conversation = cleanText(req.body?.conversation, 6000);

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
        (background ? `Candidate background (from their introduction): ${background}\n\n` : '') +
        `Category: ${category}\n` +
        `Current attempt: ${attempt} (maximum clarifications allowed: ${maxAttempts})\n\n` +
        `Question: ${question}\n\n` +
        (conversation ? `Conversation so far in this area:\n${conversation}\n\n` : '') +
        `Candidate's answer (all attempts combined): ${answer}\n\n` +
        `Reminder: speak directly to the candidate as "you" (never "the candidate", "he", "she" or "they").`,
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
// 1b) Tailored question for each area, based on the candidate's background
// ============================================

const AREA_FOCUS = {
  Product: 'building, prioritising and shipping products; deciding what to build and what to leave out',
  Business: 'commercial thinking: revenue, growth, pricing, unit economics and operations',
  'AI Fluency': 'using AI tools in real work, getting the best out of them and judging when output cannot be trusted',
  Technology: 'working with engineers and making sound technical decisions as a non-specialist or specialist',
  Marketing: 'positioning, channels, messaging and launching to an audience',
  Sales: 'understanding buyers, handling objections, discovery and closing deals',
};

const PLAN_SCHEMA = {
  name: 'tailored_question',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    required: ['question'],
    properties: {
      question: { type: 'string', description: 'One interview question spoken directly to the candidate ("you"/"your"), max 45 words, plain spoken text.' },
    },
  },
};

const PLAN_PROMPT = `You are Ava, an AI interviewer for School of Operators (SOO).
Write ONE interview question for the given area, tailored to this specific candidate.
- You are speaking directly TO the candidate. Address them as "you" and "your" ("Tell me about a time you..."),
  never in the third person ("the candidate", "he", "she", "they").
- Use their background: role, industry, experience level and what they want next. Anchor the question in their world
  (a product, team, tool or situation they mentioned) so it never feels generic.
- If they have little direct experience in this area, ask about a realistic situation from their own background or a transferable
  experience (a project, internship, club, side project) instead of a textbook question.
- Match the difficulty to their experience level. An early-career person gets practical, scenario-based questions, not senior strategy.
- Ask what they did or what they would do. Never ask for definitions, and do not assume they know jargon; use plain words.
- Do not repeat earlier questions. You may lightly build on earlier answers, but keep the focus on the area.
- Max 45 words. It is spoken aloud: plain text, no preamble such as "Next question", no lists, no emojis, exactly one question.`;

app.post('/api/plan-question', aiLimiter, async (req, res, next) => {
  try {
    const category = cleanText(req.body?.category, 100);
    const background = cleanText(req.body?.background, 1500);
    const name = cleanText(req.body?.candidateName, 100);
    if (!category) return res.status(400).json({ success: false, message: '"category" is required.' });
    const previous = Array.isArray(req.body?.previous) ? req.body.previous.slice(-4) : [];
    const prev = previous
      .map((p) => `[${cleanText(p && p.category, 60)}] Q: ${cleanText(p && p.question, 400)}\nA: ${cleanText(p && p.answer, 700)}`)
      .join('\n\n');

    const { data, model } = await askOpenAIForJson({
      system: PLAN_PROMPT,
      user:
        `Area: ${category} (${AREA_FOCUS[category] || 'general professional skills'})\n` +
        (name ? `Candidate first name (for your information only, do not use it as the subject of the question): ${name}\n` : '') +
        `Candidate background (from their introduction): ${background || 'not given'}\n` +
        (prev ? `\nEarlier questions and answers:\n${prev}\n` : '') +
        `\nWrite the question directly to the candidate, using "you".`,
      schema: PLAN_SCHEMA,
      temperature: 0.7,
      maxTokens: 150,
    });
    const question = cleanText(data.question, 400);
    if (!question) throw httpError(502, 'The AI returned an empty question.');
    return res.status(200).json({ success: true, model, question });
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
      turns: Array.isArray(item.turns)
        ? item.turns.slice(0, 14).map((t) => ({ role: t && t.role === 'ava' ? 'ava' : 'candidate', text: cleanText(t && t.text, 2500) })).filter((t) => t.text)
        : [],
    });
  }

  const name = candidate && typeof candidate.name === 'string' ? candidate.name.trim().slice(0, 100) : '';

  return { value: { candidateName: name, responses: cleaned } };
}

// The cohort's value propositions. The analysis maps each gap to the ones that genuinely address it.
const VALUE_PROPS = [
  { id: 1, title: 'A clear path into startup roles', detail: 'For people who want to work in a startup but do not know how to get in. Positions graduates for roles like Founder Office, Product Manager, Program Manager, Category Lead, P&L Lead and GTM Lead.' },
  { id: 2, title: 'Learn by doing, not just listening', detail: 'Every concept is paired with a real startup problem solved as you go, so you build actual work rather than just theory.' },
  { id: 3, title: 'Structured 10-week curriculum across the startup lifecycle', detail: 'Ten concepts over ten weeks: Pre-PMF, PMF, Post-PMF scale, GTM, roadmaps and artefacts, UX and communication, MVP, and interview prep.' },
  { id: 4, title: 'Live mentorship from an experienced operator', detail: 'Weekly live sessions with Kushagra Mohan (Head of Product at Loylty Rewardz, ex-Leap, founding PM at CasaOne and Multibhashi).' },
  { id: 5, title: 'A portfolio piece, not just a certificate', detail: 'You produce tangible artefacts (problem brief, user research, GTM proposal, product brief, UX walkthrough, MVP, final presentation) that founders and engineers actually use.' },
  { id: 6, title: 'Direct exposure and networking with startups', detail: 'You work on a real startup\'s problem statement and present your solution to that startup, which often becomes an introduction to a potential employer.' },
  { id: 7, title: 'Training for the roles that "run things"', detail: 'Explicitly for people who want to run things at a startup, not just work at one.' },
  { id: 8, title: 'Selective, hands-on, and low-commitment to explore', detail: 'A selective, hands-on programme with a free 20-minute discovery call to decide whether it is the right move before committing.' },
  { id: 9, title: 'Interview and positioning support', detail: 'Interview prep, positioning your work, and founder intros to help convert the experience into a job.' },
  { id: 10, title: 'Accessible price point', detail: 'Priced at ₹30,000 with limited seats, positioned as an investable step into startup careers.' },
];
const VALUE_PROP_BY_ID = new Map(VALUE_PROPS.map((v) => [v.id, v]));

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
      'profile_snapshot',
      'development_plan',
    ],
    properties: {
      overall_score: { type: 'number', description: 'Overall score from 1 to 10 (recalculated by the server from the category scores)' },
      profile_snapshot: { type: 'string', description: 'One or two sentences addressed to the candidate ("you") on who they are and what they want next, in their own terms' },
      development_plan: {
        type: 'object',
        additionalProperties: false,
        required: ['before_cohort', 'during_cohort'],
        properties: {
          before_cohort: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['focus', 'action', 'why'], properties: { focus: { type: 'string' }, action: { type: 'string' }, why: { type: 'string' } } } },
          during_cohort: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['focus', 'action', 'why', 'value_prop_id'],
              properties: {
                focus: { type: 'string' },
                action: { type: 'string' },
                why: { type: 'string' },
                value_prop_id: { type: 'integer', enum: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], description: 'The cohort value proposition (1-10) this action draws on' },
              },
            },
          },
        },
      },
      summary: { type: 'string', description: '2-3 sentence overall assessment, written directly to the candidate in second person ("you")' },
      strengths: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['area', 'evidence'],
          properties: {
            area: { type: 'string' },
            evidence: { type: 'string', description: 'What in your answers shows this strength (second person)' },
          },
        },
      },
      weaknesses: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['area', 'evidence', 'suggestion', 'cohort_support'],
          properties: {
            area: { type: 'string' },
            evidence: { type: 'string', description: 'What in your answers shows this gap (second person)' },
            suggestion: { type: 'string', description: 'A concrete way for you to improve (second person)' },
            cohort_support: {
              type: 'array',
              description: '0-2 cohort value propositions that genuinely address this gap. Empty array if none fits honestly.',
              items: {
                type: 'object',
                additionalProperties: false,
                required: ['value_prop_id', 'how_it_helps'],
                properties: {
                  value_prop_id: { type: 'integer', enum: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] },
                  how_it_helps: { type: 'string', description: 'One or two sentences, in second person, on how this part of the cohort closes this specific gap' },
                },
              },
            },
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
            comment: { type: 'string', description: 'Why this score, written to the candidate in second person ("you")' },
          },
        },
      },
      recommended_company_types: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['type', 'reason', 'based_on'],
          properties: {
            type: { type: 'string' },
            reason: { type: 'string', description: 'Why this fits you (second person)' },
            based_on: { type: 'string', description: 'What you said (background, goal or a specific answer) that supports this recommendation, in second person' },
          },
        },
      },
    },
  },
};

const ANALYSIS_PROMPT = `You are an expert interview coach and hiring consultant for School of Operators (SOO).
You will receive a candidate's interview transcript. The first item is their introduction (background, current role, what they want next).
Each later item is one area (Product, Business, AI Fluency, Technology, Marketing, Sales) with the main question, the candidate's answers and
any follow-up questions the interviewer asked. Answers were usually spoken and transcribed automatically, so do not penalise minor
transcription slips, filler words or missing punctuation.

VOICE (very important)
- The report is shown directly to the candidate, so write EVERYTHING in it TO them, in second person: "you", "your".
  Examples: "You gave a clear example of prioritising by impact.", "Your answer on pricing stayed fairly general.",
  "Practise quantifying the results you achieve."
- NEVER write about them in the third person. Do not use "the candidate", "the interviewee", "he", "she", "they", "them", "their",
  and do not use their name as the subject of a sentence (wrong: "Sam showed strong product sense"; right: "You showed strong product sense").
- The transcript labels the speakers "Interviewer" and "Candidate" for your reference only. Do not copy those labels into the report.
- Write "the interviewer" only when you really must refer to Ava's follow-up questions, otherwise use "I" or just describe what you said.

Analyze ONLY what is in the transcript. Do not invent experience the candidate did not mention.
Be specific, honest and constructive, and cite concrete details from the answers and follow-ups as evidence.

SCORING
- Score each area from 1 to 5 on demonstrated practical ability and quality of reasoning across the whole exchange (answer plus follow-ups),
  not on vocabulary. A candidate who does not know a term (for example RAG) but describes sound, effective practice should score well.
- Weigh the candidate's experience level: judge against what is reasonable for their background, but stay honest.
- Each category comment must say what in the transcript justifies the score, so a reviewer can check it against the transcript.
- Do not score the "Introduction" as a category. Interviewer flags that an answer did not make sense mean the answer was weak.
- overall_score will be recalculated from your category scores, so keep it consistent with them (average of the category scores x 2).

OTHER FIELDS
- profile_snapshot: one or two sentences telling the candidate who they are and what they want next ("You are a ... looking for ..."),
  using their own words where possible.
- 2-5 strengths and 2-5 weaknesses (weaknesses include an actionable suggestion).
- recommended_company_types: 2-4 types of companies or roles to target. Each must be linked to what the candidate actually said: put their
  background, stated goal or a specific answer in "based_on", and avoid generic advice that would fit anyone.
- development_plan: 2-4 items for before_cohort (what to prepare, practise or build before the cohort starts) and 2-4 items for during_cohort
  (what to focus on and practise while on the cohort). Each item has a short "focus", a concrete "action" they can do, and "why" it matters,
  tied to gaps or goals seen in their answers. Phrase each action as advice to them ("Write up two case studies of...").

COHORT MAPPING (make the report actionable)
The candidate is being considered for the School of Operators cohort. These are its value propositions:
${VALUE_PROPS.map((v) => `${v.id}. ${v.title}: ${v.detail}`).join('\n')}

- For EVERY weakness, fill "cohort_support" with 0-2 value propositions (by id) that genuinely address that specific gap, and in
  "how_it_helps" explain, in second person, how that part of the cohort closes the gap ("Weekly sessions with an operator will let you
  pressure-test your prioritisation calls", "The GTM proposal artefact gives you a place to practise the positioning you struggled with").
  Tie each mapping to the actual gap and to the week, artefact or session that fits it. Do not just repeat the value proposition.
- Typical fits: thin product or business answers -> 2, 3, 5; weak storytelling, positioning or interviewing -> 5, 9; lack of real-world
  exposure or network -> 6; unclear on how to break into startups or which role to aim for -> 1, 7; needs feedback and judgement -> 4.
  Value propositions 8 and 10 are about commitment and price, so use them only when the transcript raises doubt, cost or readiness.
- Only claim what the value propositions actually say. Never invent features, outcomes, hiring guarantees, mentors, companies or numbers.
  If no value proposition honestly fits a gap, return an empty "cohort_support" array rather than forcing a match.
- Every during_cohort item must set "value_prop_id" to the one value proposition it draws on, and its "action" should name the concrete
  part of the programme (for example the relevant artefact, a weekly session, or the startup problem statement).
- Across the whole report, spread the mappings sensibly; do not map every gap to the same value proposition.

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
        const body = r.turns && r.turns.length
          ? r.turns.map((t) => `${t.role === 'ava' ? 'Interviewer' : 'Candidate'}: ${t.text}`).join('\n')
          : `Question: ${r.question}\nAnswer: ${r.answer}`;
        return `#${i + 1} [${r.category}]\n${body}${flag}${note}`;
      })
      .join('\n\n');

    const user =
      (value.candidateName ? `Candidate's first name (use it at most once, as a direct address): ${value.candidateName}\n\n` : '') +
      `Interview transcript:\n\n${transcript}\n\n` +
      `Reminder: write the whole report directly to the candidate in second person ("you", "your"), never in the third person.`;

    const { data, model } = await askOpenAIForJson({
      system: ANALYSIS_PROMPT,
      user,
      schema: ANALYSIS_SCHEMA,
      temperature: 0.3,
    });

    // Overall score is always the average of the area scores x 2 (so 27/30 = 9.0), never a separate guess.
    const nums = (data.category_scores || []).map((c) => Number(c.score)).filter((n) => Number.isFinite(n));
    if (nums.length) {
      data.overall_score = Math.round((nums.reduce((a, n) => a + n, 0) / (nums.length * 5)) * 100) / 10;
    }

    // Attach the real value proposition titles (never trust the model to spell them) and drop invalid or duplicate ids.
    for (const w of data.weaknesses || []) {
      const seen = new Set();
      w.cohort_support = (Array.isArray(w.cohort_support) ? w.cohort_support : [])
        .filter((c) => VALUE_PROP_BY_ID.has(Number(c.value_prop_id)) && !seen.has(Number(c.value_prop_id)) && seen.add(Number(c.value_prop_id)))
        .slice(0, 2)
        .map((c) => ({
          value_prop_id: Number(c.value_prop_id),
          value_prop_title: VALUE_PROP_BY_ID.get(Number(c.value_prop_id)).title,
          how_it_helps: cleanText(c.how_it_helps, 600),
        }));
    }
    const dp = data.development_plan || {};
    for (const item of dp.during_cohort || []) {
      const vp = VALUE_PROP_BY_ID.get(Number(item.value_prop_id));
      item.value_prop_title = vp ? vp.title : '';
    }

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