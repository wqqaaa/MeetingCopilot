/**
 * Cloud ASR engine (PLAN §6.5 "Plan B"): send each VAD-endpointed segment to
 * a cloud audio-in endpoint. Two wire protocols, picked by model family:
 *
 *  - OpenAI-compatible audio chat (default): MiMo `mimo-v2.5-asr`
 *    (China-direct, ~1.1 s for 10 s audio, auto language detection).
 *  - DashScope native multimodal-generation: Alibaba `qwen-audio-3.0-asr-flash`
 *    (Token Plan-eligible; format/sample_rate live in `parameters`, response is
 *    `output.text` / `output.output.sentence.text`, NOT `choices`). Verified
 *    live 2026-10-07: 11.2 s zh audio → 1.3 s, exact transcript.
 *
 * Same interface as WhisperEngine so the worker swaps engines with no other
 * change. Runs in the utilityProcess (global fetch, direct — both endpoints are
 * China-direct). The local VAD stays the gate: only speech leaves the machine.
 */
import { encodeWav } from './wav';
import type { AsrEngine, TranscribeResult } from './engine';

export interface CloudAsrConfig {
  baseUrl: string;
  model: string;
  apiKey: string;
}

/** qwen-audio ASR models speak the DashScope native protocol, not OpenAI chat. */
export function isDashscopeAsrModel(model: string): boolean {
  return model.startsWith('qwen-audio');
}

/** Accept bare host, /v1 or /compatible-mode/v1 — all resolve to the host root. */
export function dashscopeHostRoot(baseUrl: string): string {
  return baseUrl
    .trim()
    .replace(/\/+$/, '')
    .replace(/\/compatible-mode\/v1$/i, '')
    .replace(/\/v1$/i, '');
}

export class CloudAsrEngine implements AsrEngine {
  readonly ep = 'cloud';
  readonly loadMs = 0;
  warmMs = 0;
  readonly lidAvailable = false;

  private readonly dashscopeNative: boolean;

  private constructor(private readonly cfg: CloudAsrConfig) {
    this.dashscopeNative = isDashscopeAsrModel(cfg.model);
  }

  static async load(cfg: CloudAsrConfig): Promise<CloudAsrEngine> {
    if (!cfg.baseUrl || !cfg.model || !cfg.apiKey) {
      throw new Error('cloud ASR is not configured (needs a base URL, a model and an API key)');
    }
    return new CloudAsrEngine(cfg);
  }

  /** No GPU to warm; cloud is stateless per request. */
  async warmup(): Promise<number> {
    return 0;
  }

  async transcribe(pcm: Float32Array, _language: 'auto' | string): Promise<TranscribeResult> {
    const b64 = Buffer.from(encodeWav(pcm, 16000)).toString('base64');
    const t0 = Date.now();
    const text = this.dashscopeNative
      ? await this.transcribeDashscope(b64)
      : await this.transcribeOpenAiChat(b64);
    return { text, lang: undefined, inferMs: Date.now() - t0 };
  }

  /** OpenAI-compatible audio-in chat (MiMo segment ASR). */
  private async transcribeOpenAiChat(b64: string): Promise<string> {
    const url = `${this.cfg.baseUrl.replace(/\/+$/, '')}/chat/completions`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.cfg.apiKey}` },
      body: JSON.stringify({
        model: this.cfg.model,
        messages: [{ role: 'user', content: [{ type: 'input_audio', input_audio: { data: b64, format: 'wav' } }] }],
        stream: false,
      }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`cloud ASR HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const j = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    return (j?.choices?.[0]?.message?.content ?? '').trim();
  }

  /**
   * DashScope native multimodal-generation (qwen-audio-3.0-asr-flash).
   * `format`/`sample_rate` belong in `parameters`; audio goes in as a
   * base64 Data URL. Response carries `output.text` (and per-sentence detail
   * under `output.output.sentence`), no `choices`.
   */
  private async transcribeDashscope(b64: string): Promise<string> {
    const url = `${dashscopeHostRoot(this.cfg.baseUrl)}/api/v1/services/aigc/multimodal-generation/generation`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.cfg.apiKey}`,
        'X-DashScope-SSE': 'disable',
      },
      body: JSON.stringify({
        model: this.cfg.model,
        input: {
          messages: [
            {
              role: 'user',
              content: [{ type: 'input_audio', input_audio: { data: `data:audio/wav;base64,${b64}` } }],
            },
          ],
        },
        parameters: { format: 'wav', sample_rate: 16000 },
      }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`cloud ASR HTTP ${res.status}: ${body.slice(0, 200)}`);
    }
    const j = (await res.json()) as {
      output?: { text?: string; output?: { sentence?: { text?: string } } };
    };
    return (j?.output?.text ?? j?.output?.output?.sentence?.text ?? '').trim();
  }
}
