/**
 * Financial Advisor Agent - TypeScript implementation mirroring Python/Go agent architecture.
 *
 * This agent provides financial advisory capabilities using the LLP TypeScript SDK.
 * Structure mirrors the Python financial_advisor and Go devops_agent examples.
 */

import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Load .env from the same directory as this script
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: join(__dirname, '.env') });

import { LLPClient, TextMessage, LLPClientConfig } from 'llpsdk';
import { ChatOllama } from '@langchain/ollama';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { WebPDFLoader } from '@langchain/community/document_loaders/web/pdf';

// =============================================================================
// Constants
// =============================================================================

const SYSTEM_PROMPT = `
You are a Financial Advisor AI assistant that analyzes financial questions and provides guidance.
You MUST return ONLY valid JSON responses.

## Response Format

Return JSON matching ONE of these formats:

### For Financial Analysis Questions:
{
    "type": "analysis",
    "category": "investment" | "budgeting" | "retirement" | "tax" | "debt" | "savings" | "invoices",
    "risk_level": "low" | "medium" | "high",
    "recommendation": "Your specific evaluation",
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
- Investment strategies and portfolio allocation
- Budgeting and expense management
- Retirement planning (401k, IRA, pensions)
- Tax optimization strategies
- Debt management and payoff strategies
- Emergency fund and savings goals
- Risk assessment and management
- Analysis of invoices as PDF files

## Important Rules:
1. NEVER provide specific stock picks or guarantees
2. Always recommend consulting a licensed financial advisor for major decisions
3. Consider the user's risk tolerance when applicable
4. Provide educational information, not personalized financial advice
5. Be clear about limitations and uncertainties

## Category Definitions:
- investment: Questions about stocks, bonds, ETFs, portfolio allocation
- budgeting: Questions about spending, income management, expense tracking
- retirement: Questions about 401k, IRA, pension, retirement age planning
- tax: Questions about tax strategies, deductions, tax-advantaged accounts
- debt: Questions about loans, credit cards, debt payoff strategies
- savings: Questions about emergency funds, savings goals, high-yield accounts
- invoices: Questions about invoice details, recite details about the invoice
`;

// =============================================================================
// Types (mirrors Python dataclasses / Go structs)
// =============================================================================

interface FinancialAnalysis {
	type: 'analysis' | 'capabilities' | 'decline' | string;
	category?: 'investment' | 'budgeting' | 'retirement' | 'tax' | 'debt' | 'savings';
	risk_level?: 'low' | 'medium' | 'high';
	recommendation?: string;
	considerations?: string[];
	reason?: string; // For decline responses
}

// =============================================================================
// LangChain setup
// =============================================================================

const outputParser = new JsonOutputParser<FinancialAnalysis>();

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Return capabilities description.
 */
function formatCapabilities(): string {
	return `I'm a Financial Advisor assistant. I can help with:

* Investment Strategies - Portfolio allocation, diversification, risk management
* Budgeting - Expense tracking, income management, spending plans
* Retirement Planning - 401(k), IRA, pension strategies
* Tax Optimization - Tax-advantaged accounts, deduction strategies
* Debt Management - Payoff strategies, refinancing options
* Savings Goals - Emergency funds, high-yield accounts

Note: I provide educational information, not personalized financial advice.
Always consult a licensed financial advisor for major financial decisions.`;
}

/**
 * Format analysis response for user.
 */
function formatAnalysis(analysis: FinancialAnalysis): string {
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
// Message Handler (mirrors Python/Go handleMessage pattern)
// =============================================================================

export async function handleMessage(llm: ChatOllama, message: TextMessage): Promise<string> {
	// Use message ID as correlation ID for tracing request/response pairs
	const corrId = message.id?.slice(0, 8) ?? 'no-id';
	const preview = message.prompt.slice(0, 80);
	console.log(`[${corrId}] >>> RECV from=${message.sender} prompt="${preview}"`);
	console.log(message);

	// Fetch PDF attachment via WebPDFLoader
	let attachmentContent = '';
	if (message.hasAttachment()) {
		try {
			console.log(`[${corrId}] ... fetching attachment url=${message.attachment}`);
			const response = await fetch(message.attachment);
			if (!response.ok) {
				console.warn(`[${corrId}] !!! ATTACHMENT FETCH FAILED status=${response.status}`);
			} else {
				const blob = await response.blob();
				const loader = new WebPDFLoader(blob, {
					splitPages: false,
					pdfjs: () => import('pdfjs-dist/legacy/build/pdf.mjs') as never,
				});
				const docs = await loader.load();
				attachmentContent = docs.map(d => d.pageContent).join('\n');
				console.log(`[${corrId}] <<< attachment parsed pages=${docs.length} len=${attachmentContent.length}`);
			}
		} catch (err) {
			console.error(`[${corrId}] !!! ATTACHMENT FETCH ERROR error=${err}`);
		}
	}

	// Build prompt with attachment content if available
	const fullPrompt = attachmentContent
		? `${message.prompt}\n\nAttachment content type: application/pdf\nAttachment content:\n${attachmentContent}`
		: message.prompt;

	// Call LLM via LangChain chain: messages → ChatOllama → JsonOutputParser
	let analysis: FinancialAnalysis;
	try {
		console.log(`[${corrId}] ... calling LLM`);
		const chain = llm.pipe(outputParser);
		analysis = await chain.invoke([
			new SystemMessage(SYSTEM_PROMPT),
			new HumanMessage(fullPrompt),
		]);
		console.log(`[${corrId}] === type=${analysis.type} category=${analysis.category ?? 'n/a'}`);
	} catch (err) {
		console.error(`[${corrId}] !!! LLM CALL FAILED error=${err}`);
		return "I'm sorry, I encountered an error processing your request.";
	}

	// Route based on response type
	if (analysis.type === 'capabilities') {
		return formatCapabilities();
	} else if (analysis.type === 'decline') {
		return analysis.reason ?? 'I can only help with financial questions.';
	} else if (analysis.type === 'analysis') {
		return formatAnalysis(analysis);
	} else {
		return JSON.stringify(analysis);
	}
}

// =============================================================================
// Main Entry Point (mirrors Python/Go agent structure)
// =============================================================================

async function main(): Promise<void> {
	// 1. Load environment variables
	const agentName = process.env.AGENT_NAME ?? 'financial-advisor-ts';
	const apiKey = process.env.AGENT_KEY ?? 'Z22MAsvpGFrMMX9qLZZqIynKp/42gBa4Edl/X94MFkA';

	// 2. Initialize LangChain ChatOllama client
	const model = process.env.OLLAMA_MODEL ?? 'gpt-oss:120b';
	const llm = new ChatOllama({
		baseUrl: process.env.OLLAMA_HOST ?? 'http://localhost:11434',
		model,
		headers: process.env.OLLAMA_API_KEY
			? { Authorization: `Bearer ${process.env.OLLAMA_API_KEY}` }
			: undefined,
	});
	console.log(`LangChain ChatOllama initialized model=${model}`);

	// 3. Initialize LLP client
	const llpConfig: LLPClientConfig = {
		url: process.env.PLATFORM_ADDRESS ?? 'ws://localhost:4000/agent/websocket',
		responseTimeout: 600000, // 10 minutes
	};
	const client = new LLPClient(agentName, apiKey, llpConfig);

	client.onMessage(async (msg: TextMessage) => {
		const corrId = msg.id?.slice(0, 8) ?? 'no-id';
		console.log(`[${corrId}] --- REQUEST START ---`);
		const response = await handleMessage(llm, msg);
		console.log(`[${corrId}] <<< SEND to=${msg.sender} len=${response.length}`);
		console.log(`[${corrId}] --- REQUEST END ---`);
		return msg.reply(response);
	});

	// 5. Setup graceful shutdown
	const shutdown = async () => {
		console.log('\nShutting down...');
		await client.close();
		console.log('Disconnected');
		process.exit(0);
	};

	process.on('SIGINT', shutdown);
	process.on('SIGTERM', shutdown);

	// 6. Connect and run
	try {
		await client.connect();
		console.log('Connected to platform');

		// Wait forever
		await new Promise(() => {});
	} catch (err) {
		console.error('Fatal error:', err);
		process.exit(1);
	}
}

main();
