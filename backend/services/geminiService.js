import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
dotenv.config();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── System prompt builder ────────────────────────────────────────────────────
function buildSystemInstruction(profile) {
  return `You are LearnAI, an expert personalized learning advisor with deep knowledge across all domains — technology, business, design, science, arts, mathematics, languages, and more.

Your mission is to help ${profile.name} achieve their learning goals through thoughtful, personalized guidance.

LEARNER PROFILE:
━━━━━━━━━━━━━━━
• Name: ${profile.name}
• Experience Level: ${profile.experienceLevel || 'beginner'}
• Interests: ${(profile.interests || []).join(', ') || 'Not yet specified'}
• Goals: ${(profile.goals || []).join(', ') || 'Not yet specified'}
• Learning Style: ${profile.learningStyle || 'mixed'}
• Weekly Hours Available: ${profile.weeklyHours || 5} hours/week
• Completed Courses: ${(profile.completedCourses || []).join(', ') || 'None yet'}
• Current Skills: ${(profile.skills || []).map((s) => `${s.name} (${s.level}%)`).join(', ') || 'Not specified'}

BEHAVIOR GUIDELINES:
━━━━━━━━━━━━━━━━━━━
1. Be warm, encouraging, and supportive — learning is a journey, not a race
2. Tailor language complexity to the learner's experience level
3. When goals are vague, ask 1-2 focused clarifying questions
4. Suggest specific, named resources with brief explanations
5. Reference their profile details to make responses feel personalized
6. Keep responses focused: 2-4 paragraphs max

SPECIAL ACTIONS (emit these tags when appropriate):
• When the user is ready to generate/update their learning path, emit: [ACTION:GENERATE_ROADMAP]
• When you detect profile information in conversation, emit: [ACTION:UPDATE_PROFILE:{"field":"value"}]
  Supported fields: experienceLevel, learningStyle, weeklyHours, interests (array), goals (array)

Examples:
- User says "I'm a total beginner" → emit [ACTION:UPDATE_PROFILE:{"experienceLevel":"beginner"}]
- User says "I want to learn web development" → emit [ACTION:UPDATE_PROFILE:{"goals":["Master web development"]}]
- User says "Generate my learning path" → emit [ACTION:GENERATE_ROADMAP]`;
}

// ─── Multi-turn chat ──────────────────────────────────────────────────────────
export async function chatWithContext(messages, profile) {
  const systemInstruction = buildSystemInstruction(profile);

  // Convert messages to Gemini format
  const contents = messages.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents,
    config: {
      systemInstruction,
      temperature: 0.7,
      maxOutputTokens: 1024,
    },
  });

  const rawText = response.text || '';

  // Parse embedded actions
  const actions = [];

  if (/\[ACTION:GENERATE_ROADMAP\]/.test(rawText)) {
    actions.push({ type: 'GENERATE_ROADMAP' });
  }

  const updateMatches = [...rawText.matchAll(/\[ACTION:UPDATE_PROFILE:(\{.*?\})\]/g)];
  for (const match of updateMatches) {
    try {
      actions.push({ type: 'UPDATE_PROFILE', data: JSON.parse(match[1]) });
    } catch {
      // malformed JSON — skip
    }
  }

  // Strip action tags from displayed text
  const cleanText = rawText
    .replace(/\[ACTION:GENERATE_ROADMAP\]/g, '')
    .replace(/\[ACTION:UPDATE_PROFILE:\{.*?\}\]/g, '')
    .trim();

  return { reply: cleanText, actions };
}

// ─── Learning path generator ──────────────────────────────────────────────────
export async function generateLearningPath(goal, profile) {
  const prompt = `You are a world-class curriculum designer. Create a comprehensive, personalized learning path.

LEARNER PROFILE:
• Name: ${profile.name}
• Experience Level: ${profile.experienceLevel || 'beginner'}
• Current Skills: ${(profile.skills || []).map((s) => s.name).join(', ') || 'None specified'}
• Completed Courses: ${(profile.completedCourses || []).join(', ') || 'None'}
• Learning Style: ${profile.learningStyle || 'mixed'}
• Weekly Hours: ${profile.weeklyHours || 5} hours/week
• Interests: ${(profile.interests || []).join(', ') || 'General'}

LEARNING GOAL: ${goal || (profile.goals || []).join('; ') || 'Become a well-rounded professional'}

Generate a structured, progressive learning path. Return ONLY valid JSON matching this schema exactly:

{
  "goal": "Clear, specific goal statement",
  "totalDuration": "X months",
  "estimatedCompletion": "Month YYYY",
  "skillsToGain": ["skill1", "skill2", "skill3"],
  "prerequisites": ["prereq1"],
  "phases": [
    {
      "id": "phase-1",
      "title": "Phase Title",
      "description": "What this phase covers and why",
      "order": 1,
      "skills": ["skill1", "skill2"],
      "milestones": [
        {
          "id": "m-1-1",
          "title": "Milestone Title",
          "description": "Concrete achievement to unlock",
          "resources": [
            {
              "id": "r-1-1-1",
              "title": "Exact resource title",
              "type": "course",
              "url": "https://real-platform.com/course",
              "duration": "X hours",
              "difficulty": "beginner",
              "skills": ["skill"],
              "description": "What you will learn in this resource",
              "whyRecommended": "2-3 sentences specifically explaining why this is perfect for ${profile.name} given their background and goal"
            }
          ]
        }
      ]
    }
  ]
}

REQUIREMENTS:
1. Create 3-5 phases (beginner → intermediate → advanced progression)
2. Each phase: 2-3 milestones
3. Each milestone: 2-4 resources (mix types: course, project, article, video)
4. Use REAL platforms: Coursera, Udemy, freeCodeCamp, Khan Academy, YouTube, MDN, The Odin Project, Codecademy, edX, Pluralsight, Scrimba, CS50, fast.ai, etc.
5. Use REAL URLs to actual existing courses
6. Make the whyRecommended truly personal to this learner's profile
7. Adapt difficulty/pace to ${profile.weeklyHours || 5} hours/week
8. Include at least one hands-on project per phase
9. Build prerequisites correctly (each phase builds on previous)

Return ONLY the JSON. No markdown fences, no explanation.`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      temperature: 0.3,
      maxOutputTokens: 8192,
      responseMimeType: 'application/json',
    },
  });

  try {
    return JSON.parse(response.text);
  } catch {
    // Fallback: extract JSON block if extra text leaked in
    const match = (response.text || '').match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw new Error('Failed to parse learning path from Gemini response');
  }
}

// ─── Skill gap analysis ───────────────────────────────────────────────────────
export async function analyzeSkillGaps(profile, goal) {
  const prompt = `Analyze skill gaps for a learner wanting to achieve: "${goal}"

Current State:
- Experience Level: ${profile.experienceLevel || 'beginner'}
- Known Skills: ${(profile.skills || []).map((s) => s.name).join(', ') || 'None'}
- Completed Courses: ${(profile.completedCourses || []).join(', ') || 'None'}

Return JSON:
{
  "gaps": [
    { "skill": "name", "priority": "high|medium|low", "reason": "why this skill is needed for the goal" }
  ],
  "strengths": ["existing strength to leverage"],
  "estimatedTimeToGoal": "X months"
}`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      temperature: 0.2,
      maxOutputTokens: 1024,
      responseMimeType: 'application/json',
    },
  });

  try {
    return JSON.parse(response.text);
  } catch {
    return { gaps: [], strengths: [], estimatedTimeToGoal: 'Unknown' };
  }
}

// ─── Explain a single recommendation ─────────────────────────────────────────
export async function explainRecommendation(resource, profile) {
  const prompt = `In 3 sentences, explain to ${profile.name} (${profile.experienceLevel || 'beginner'} level) 
why "${resource.title}" is specifically recommended for them.

Their goal: ${(profile.goals || []).join(', ')}
Their interests: ${(profile.interests || []).join(', ')}
Their learning style: ${profile.learningStyle || 'mixed'}

Be personal, specific, and encouraging. Avoid generic statements.`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { temperature: 0.6, maxOutputTokens: 256 },
  });

  return response.text || '';
}
