/**
 * Loan Advisor Agent — Mastra implementation.
 *
 * Mirrors financial-advisor.ts in structure and complexity.
 * Differences: Mastra instead of LangChain, SearXNG web search instead of PDF tool.
 */

import { config } from 'dotenv';
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: join(__dirname, '.env') });

import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { RequestContext } from '@mastra/core/request-context';
import { LLPClient, TextMessage, type LLPClientConfig, type Annotater } from 'llpsdk';
import { wrapWithLLPAnnotation, type LLPMastraContext } from 'llpsdk/mastra';
import * as z from 'zod';

// =============================================================================
// System Prompt
// =============================================================================

const SYSTEM_PROMPT = `
You are a Loan Advisor AI assistant that answers questions about different types of loans.
You MUST return ONLY valid JSON responses.

## Response Format

Return JSON matching ONE of these formats:

### For Loan Analysis Questions:
{
    "type": "analysis",
    "category": "home" | "auto" | "student" | "personal" | "general",
    "risk_level": "low" | "medium" | "high",
    "recommendation": "Your specific guidance",
    "considerations": ["factor1", "factor2", ...]
}

### For Capabilities Questions:
{
    "type": "capabilities"
}

### For Out-of-Domain Questions:
{
    "type": "decline",
    "reason": "Polite explanation of why you cannot help"
}

## Your Expertise Areas:
- Home loans: mortgages, fixed vs ARM, refinancing, PMI, FHA/VA/conventional loans
- Auto loans: new vs used, loan terms, dealer financing vs credit unions
- Student loans: federal vs private, repayment plans, income-driven options, refinancing
- Personal loans: unsecured loans, debt consolidation, credit score requirements
- Current interest rates: use the get_loan_rates tool when the user asks about rates

## Important Rules:
1. Use the get_loan_rates tool when the user asks about current or recent rates
2. NEVER guarantee specific rates, approval, or savings
3. Always recommend consulting a licensed loan officer for major decisions
4. Provide educational information, not personalized financial advice
5. State assumptions clearly and surface missing information

## Category Definitions:
- home: mortgages, refinancing, home equity, FHA/VA loans
- auto: car loans, auto refinancing, dealer financing
- student: federal/private student loans, PLUS loans, refinancing
- personal: unsecured personal loans, debt consolidation
- general: cross-category questions or rate comparisons
`;

// =============================================================================
// Types
// =============================================================================

const loanResponseSchema = z.object({
	type: z.union([z.literal('analysis'), z.literal('capabilities'), z.literal('decline'), z.string()]),
	category: z.enum(['home', 'auto', 'student', 'personal', 'general']).optional(),
	risk_level: z.enum(['low', 'medium', 'high']).optional(),
	recommendation: z.string().optional(),
	considerations: z.array(z.string()).optional(),
	reason: z.string().optional(),
});

type LoanResponse = z.infer<typeof loanResponseSchema>;

// =============================================================================
// Runtime Context
// =============================================================================

// LLPMastraContext from llpsdk/mastra provides the standard field names:
//   { llpMessage: TextMessage; llpAnnotater: Annotater }
// Used as: new RequestContext<LLPMastraContext>()

// =============================================================================
// SearXNG helpers
// =============================================================================

interface SearXNGResult {
	title: string;
	url: string;
	content: string;
}

interface ParsedRateResult {
	source: string;
	snippet: string;
}

function parseRateResults(results: SearXNGResult[]): ParsedRateResult[] {
	return results
		.slice(0, 5)
		.filter(r => r.content && r.title)
		.map(r => ({
			source: r.title.trim(),
			snippet: r.content
				.replace(/<[^>]+>/g, '')
				.replace(/\s+/g, ' ')
				.trim()
				.slice(0, 300),
		}))
		.filter(r => r.snippet.length > 0);
}

function formatRateResults(parsed: ParsedRateResult[]): string {
	if (parsed.length === 0) return 'No rate information found.';
	return parsed.map(r => `[${r.source}]\n${r.snippet}`).join('\n\n');
}

async function searchSearXNG(query: string, searxngUrl: string): Promise<string> {
	const url = `${searxngUrl}/search?q=${encodeURIComponent(query)}&format=json`;
	const res = await fetch(url);
	if (!res.ok) throw new Error(`SearXNG returned ${res.status}`);
	const data = (await res.json()) as { results: SearXNGResult[] };
	const parsed = parseRateResults(data.results ?? []);
	return formatRateResults(parsed);
}

// =============================================================================
// Tool
// =============================================================================

const getLoanRatesTool = createTool({
	id: 'get_loan_rates',
	description:
		'Search for current interest rates for a given loan type (home, auto, student, personal).',
	inputSchema: z.object({
		loan_type: z.enum(['home', 'auto', 'student', 'personal']),
	}),
	execute: wrapWithLLPAnnotation<{ loan_type: 'home' | 'auto' | 'student' | 'personal' }, string>(
		'get_loan_rates',
		async (inputData) => {
			const query = `current ${inputData.loan_type} loan interest rates today`;
			return await searchSearXNG(query, process.env.SEARXNG_URL ?? 'http://localhost:8080');
		},
	),
});

// =============================================================================
// Agent
// =============================================================================

export function createLoanAdvisorAgent(model: string) {
	return new Agent({
		id: 'loan-advisor',
		name: 'loan-advisor',
		instructions: SYSTEM_PROMPT,
		model,
		tools: { get_loan_rates: getLoanRatesTool },
	});
}

export type LoanAdvisorAgent = ReturnType<typeof createLoanAdvisorAgent>;

// =============================================================================
// Helper Functions
// =============================================================================

function formatCapabilities(): string {
	return `I'm a Loan Advisor assistant. I can help with:

* Home Loans — mortgages, fixed vs ARM, refinancing, PMI, FHA/VA loans
* Auto Loans — new vs used vehicle financing, dealer vs credit union rates
* Student Loans — federal vs private, repayment plans, refinancing options
* Personal Loans — unsecured loans, debt consolidation, credit requirements
* Current Rates — live rate lookups for any loan type

Note: I provide educational information, not personalized financial advice.
Always consult a licensed loan officer before making major financial decisions.`;
}

function formatAnalysis(analysis: LoanResponse): string {
	const parts: string[] = [];

	if (analysis.category) {
		const category = analysis.category.charAt(0).toUpperCase() + analysis.category.slice(1);
		parts.push(`Category: ${category}`);
	}

	if (analysis.risk_level) {
		const riskLevel = analysis.risk_level.charAt(0).toUpperCase() + analysis.risk_level.slice(1);
		parts.push(`Risk Level: ${riskLevel}`);
	}

	if (analysis.recommendation) {
		parts.push(`\n${analysis.recommendation}`);
	}

	if (analysis.considerations && analysis.considerations.length > 0) {
		parts.push('\nConsiderations:');
		for (const item of analysis.considerations) {
			parts.push(`  * ${item}`);
		}
	}

	parts.push('\n\nNote: This is educational information, not personalized financial advice.');

	return parts.join('\n');
}

// =============================================================================
// Prompt Builder
// =============================================================================

function buildPrompt(message: TextMessage): string {
	return `${message.prompt}

Return ONLY valid JSON matching the required schema.`;
}

// =============================================================================
// Response Parsing
// =============================================================================

function extractJson(text: string): string {
	const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
	if (codeBlock) return codeBlock[1].trim();
	return text.trim();
}

function parseResponse(text: string): LoanResponse {
	const json = extractJson(text);
	const parsed = JSON.parse(json) as unknown;
	return loanResponseSchema.parse(parsed);
}

// =============================================================================
// Message Handler
// =============================================================================

export async function handleMessage(
	agent: LoanAdvisorAgent,
	message: TextMessage,
	annotater: Annotater,
): Promise<string> {
	const corrId = message.id?.slice(0, 8) ?? 'no-id';
	const preview = message.prompt.slice(0, 80);
	console.log(`[${corrId}] >>> RECV from=${message.sender} prompt="${preview}"`);

	const requestContext = new RequestContext<LLPMastraContext>();
	requestContext.set('llpMessage', message);
	requestContext.set('llpAnnotater', annotater);

	let response: LoanResponse;
	try {
		console.log(`[${corrId}] ... calling agent`);
		const result = await agent.generate(buildPrompt(message), { requestContext });
		response = parseResponse(result.text);
		console.log(`[${corrId}] === type=${response.type} category=${response.category ?? 'n/a'}`);
	} catch (err) {
		console.error(`[${corrId}] !!! AGENT CALL FAILED error=${err}`);
		return "I'm sorry, I encountered an error processing your request.";
	}

	if (response.type === 'capabilities') {
		return formatCapabilities();
	} else if (response.type === 'decline') {
		return response.reason ?? 'I can only help with loan-related questions.';
	} else if (response.type === 'analysis') {
		return formatAnalysis(response);
	} else {
		return JSON.stringify(response);
	}
}

// =============================================================================
// Main Entry Point
// =============================================================================

async function main(): Promise<void> {
	const agentName = process.env.AGENT_NAME ?? 'loan-advisor';
	const apiKey = process.env.AGENT_KEY ?? '';

	const model = process.env.MODEL_NAME ?? 'ollama/llama3.1';
	console.log(`Mastra Agent initialized model=${model}`);

	const llpConfig: LLPClientConfig = {
		url: process.env.PLATFORM_ADDRESS ?? 'ws://localhost:4000/agent/websocket',
		responseTimeout: 600000,
	};

	const client = new LLPClient(agentName, apiKey, llpConfig)
		.onStart(() => createLoanAdvisorAgent(model))
		.onMessage(async (agent, msg, annotater) => {
			const response = await handleMessage(agent, msg, annotater);
			return msg.reply(response);
		})
		.onStop(() => {
			console.log('session ended');
		});

	const shutdown = async () => {
		console.log('\nShutting down...');
		await client.close();
		console.log('Disconnected');
		process.exit(0);
	};

	process.on('SIGINT', shutdown);
	process.on('SIGTERM', shutdown);

	try {
		await client.connect();
		console.log('Connected to platform');
		await new Promise(() => {});
	} catch (err) {
		console.error('Fatal error:', err);
		process.exit(1);
	}
}

const isMainModule =
	process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
	void main();
}
