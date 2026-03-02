/**
 * Test harness for the financial advisor LLM + PDF parsing path.
 * Bypasses the LLP platform entirely — no WebSocket connection needed.
 *
 * Usage:
 *   npx tsx test-financial-advisor.ts                          # runs text-only tests
 *   npx tsx test-financial-advisor.ts <pdf-url>               # also runs PDF test with given URL
 *
 * Examples:
 *   npx tsx test-financial-advisor.ts
 *   npx tsx test-financial-advisor.ts https://example.com/invoice.pdf
 */

import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: join(__dirname, '.env') });

import { ChatOllama } from '@langchain/ollama';
import { handleMessage } from './financial-advisor.js';

// =============================================================================
// Minimal TextMessage stub — mirrors the llpsdk TextMessage interface
// =============================================================================

function makeMessage(prompt: string, attachmentUrl?: string) {
	return {
		id: `test-${Date.now()}`,
		sender: 'test-harness',
		prompt,
		attachment: attachmentUrl ?? '',
		hasAttachment: () => !!attachmentUrl,
		reply: async (response: string) => {
			// No-op in tests — we capture the return value directly
			return response;
		},
	};
}

// =============================================================================
// Test runner
// =============================================================================

async function runTest(label: string, fn: () => Promise<void>): Promise<void> {
	console.log(`\n${'='.repeat(60)}`);
	console.log(`TEST: ${label}`);
	console.log('='.repeat(60));
	try {
		await fn();
		console.log('\n✓ PASSED');
	} catch (err) {
		console.error('\n✗ FAILED:', err);
	}
}

// =============================================================================
// Main
// =============================================================================

async function main(): Promise<void> {
	const pdfUrl = process.argv[2]; // optional CLI argument

	const llm = new ChatOllama({
		baseUrl: process.env.OLLAMA_HOST ?? 'http://localhost:11434',
		model: process.env.OLLAMA_MODEL ?? 'gpt-oss:120b',
		headers: process.env.OLLAMA_API_KEY
			? { Authorization: `Bearer ${process.env.OLLAMA_API_KEY}` }
			: undefined,
	});

	console.log(`Ollama: ${process.env.OLLAMA_HOST} / model: ${process.env.OLLAMA_MODEL}`);
	console.log(`PDF URL: ${pdfUrl ?? '(none — skipping PDF test)'}`);

	// --- Test 1: capabilities question ---
	await runTest('Capabilities question (no PDF)', async () => {
		const msg = makeMessage('What can you help me with?');
		const result = await handleMessage(llm, msg);
		console.log('\nResponse:\n', result);
		if (!result.includes('Investment')) throw new Error('Expected capabilities response');
	});

	// --- Test 2: financial analysis question ---
	await runTest('Financial analysis question (no PDF)', async () => {
		const msg = makeMessage('I have $10,000 saved. Should I pay off my credit card debt at 20% APR or invest in an index fund?');
		const result = await handleMessage(llm, msg);
		console.log('\nResponse:\n', result);
		if (!result.includes('Category') && !result.includes('Risk Level')) {
			throw new Error('Expected analysis response with Category/Risk Level');
		}
	});

	// --- Test 3: out-of-domain question ---
	await runTest('Out-of-domain question (decline)', async () => {
		const msg = makeMessage('What is the capital of France?');
		const result = await handleMessage(llm, msg);
		console.log('\nResponse:\n', result);
	});

	// --- Test 4: PDF invoice (only if URL provided) ---
	if (pdfUrl) {
		await runTest(`PDF invoice parsing — ${pdfUrl}`, async () => {
			const msg = makeMessage('Please summarize the details of this invoice.', pdfUrl);
			// Note: WebPDFLoader is exercised inside handleMessage
			const result = await handleMessage(llm, msg);
			console.log('\nResponse:\n', result);
			if (!result || result.length < 10) throw new Error('Expected non-empty response for PDF');
		});
	}

	console.log(`\n${'='.repeat(60)}`);
	console.log('All tests complete.');
}

main().catch(err => {
	console.error('Fatal:', err);
	process.exit(1);
});
