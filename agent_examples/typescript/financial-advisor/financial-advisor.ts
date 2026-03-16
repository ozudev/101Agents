/**
 * Financial Advisor Agent - TypeScript implementation mirroring Python/Go agent architecture.
 *
 * This agent provides financial advisory capabilities using the LLP TypeScript SDK.
 * Structure mirrors the Python financial_advisor and Go devops_agent examples.
 */

import { config } from 'dotenv';
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

// Load .env from the same directory as this script
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: join(__dirname, '.env') });

import { LLPClient, TextMessage, type LLPClientConfig, type Annotater } from 'llpsdk';
import { createLLPToolMiddleware } from 'llpsdk/langchain';
import { ChatOllama } from '@langchain/ollama';
import { HumanMessage } from '@langchain/core/messages';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { WebPDFLoader } from '@langchain/community/document_loaders/web/pdf';
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

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

const financialAnalysisSchema = z.object({
	type: z.union([z.literal('analysis'), z.literal('capabilities'), z.literal('decline'), z.string()]),
	category: z.enum(['investment', 'budgeting', 'retirement', 'tax', 'debt', 'savings', 'invoices']).optional(),
	risk_level: z.enum(['low', 'medium', 'high']).optional(),
	recommendation: z.string().optional(),
	considerations: z.array(z.string()).optional(),
	reason: z.string().optional(),
});

type FinancialAnalysis = z.infer<typeof financialAnalysisSchema>;

// =============================================================================
// LangChain setup
// =============================================================================

const outputParser = new JsonOutputParser<FinancialAnalysis>();

async function loadPdfText(url: string): Promise<string> {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Attachment fetch failed with status ${response.status}`);
	}

	const blob = await response.blob();
	const loader = new WebPDFLoader(blob, {
		splitPages: false,
		pdfjs: () => import('pdfjs-dist/legacy/build/pdf.mjs') as never,
	});
	const docs = await loader.load();
	return docs.map(d => d.pageContent).join('\n');
}

const loadInvoicePdfTool = tool(
	async ({ url }: { url: string }) => {
		const content = await loadPdfText(url);
		return content;
	},
	{
		name: 'load_invoice_pdf',
		description:
			'Load and extract text from a PDF invoice attachment when the user asks about invoice contents or an attached PDF.',
		schema: z.object({
			url: z.string().url().describe('The attachment URL for the PDF to inspect'),
		}),
	},
);

function extractTextContent(content: unknown): string {
	if (typeof content === 'string') {
		return content;
	}

	if (Array.isArray(content)) {
		return content
			.map(item => {
				if (typeof item === 'string') {
					return item;
				}
				if (item && typeof item === 'object' && 'text' in item) {
					return String((item as { text?: unknown }).text ?? '');
				}
				return '';
			})
			.filter(Boolean)
			.join('\n');
	}

	return String(content ?? '');
}

function buildPrompt(message: TextMessage): string {
	const attachmentUrl = message.attachment;
	if (!message.hasAttachment() || !attachmentUrl) {
		return `${message.prompt}

Return ONLY valid JSON matching the required schema.`;
	}

	return `${message.prompt}

An attachment is available at this URL: ${attachmentUrl}
If the user is asking about the attachment or an invoice, use the load_invoice_pdf tool with that URL before answering.
Return ONLY valid JSON matching the required schema.`;
}

async function parseAnalysisResponse(rawText: string): Promise<FinancialAnalysis> {
	const parsed = await outputParser.invoke(rawText);
	return financialAnalysisSchema.parse(parsed);
}

export function createFinancialAdvisorAgent(llm: ChatOllama) {
	return createAgent({
		model: llm,
		tools: [loadInvoicePdfTool],
		middleware: [createLLPToolMiddleware()],
		systemPrompt: SYSTEM_PROMPT,
	});
}

export type FinancialAdvisorAgent = ReturnType<typeof createFinancialAdvisorAgent>;

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

export async function handleMessage(
	agent: FinancialAdvisorAgent,
	message: TextMessage,
	annotater: Annotater,
): Promise<string> {
	const corrId = message.id?.slice(0, 8) ?? 'no-id';
	const preview = message.prompt.slice(0, 80);
	console.log(`[${corrId}] >>> RECV from=${message.sender} prompt="${preview}"`);

	const fullPrompt = buildPrompt(message);

	let analysis: FinancialAnalysis;
	try {
		console.log(`[${corrId}] ... calling agent`);
		const result = await agent.invoke(
			{ messages: [new HumanMessage(fullPrompt)] },
			{ context: { llpMessage: message, llpClient: annotater } },
		);
		const lastMessage = result.messages[result.messages.length - 1];
		const rawText = extractTextContent(lastMessage?.content);
		analysis = await parseAnalysisResponse(rawText);
		console.log(`[${corrId}] === type=${analysis.type} category=${analysis.category ?? 'n/a'}`);
	} catch (err) {
		console.error(`[${corrId}] !!! LLM CALL FAILED error=${err}`);
		return "I'm sorry, I encountered an error processing your request.";
	}

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
	const agentName = process.env.AGENT_NAME ?? 'financial-advisor-ts';
	const apiKey = process.env.AGENT_KEY ?? '';

	const model = process.env.OLLAMA_MODEL ?? 'gpt-oss:120b';
	const llm = new ChatOllama({
		baseUrl: process.env.OLLAMA_HOST ?? 'http://localhost:11434',
		model,
		headers: process.env.OLLAMA_API_KEY
			? { Authorization: `Bearer ${process.env.OLLAMA_API_KEY}` }
			: undefined,
	});
	console.log(`LangChain ChatOllama initialized model=${model}`);

	const llpConfig: LLPClientConfig = {
		url: process.env.PLATFORM_ADDRESS ?? 'ws://localhost:4000/agent/websocket',
		responseTimeout: 600000,
	};

	const client = new LLPClient(agentName, apiKey, llpConfig)
		.onStart(() => createFinancialAdvisorAgent(llm))
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
