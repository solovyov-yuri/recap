import { Copy, Download, FolderOpen, RotateCcw } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input, Label, Segmented, Select } from "@/components/ui/controls";
import { useToast } from "@/components/ui/toast";
import { getBridge } from "@/lib/bridge";
import type { AppSettings, ExportFormat, RunResult, SummaryMode } from "@/lib/types";
import { dirName, stem } from "@/lib/utils";
import type { Phase, RunOverrides } from "@/hooks/useRecap";

const PROVIDERS = ["openai", "xai", "ollama", "lm-studio", "vllm"].map((v) => ({ value: v, label: v }));

const TRANSCRIPTION_LANGS = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
  { value: "auto", label: "Авто" },
];

const EXPORT_LABELS: Record<ExportFormat, string> = {
  telegram: "Telegram (.txt)",
  plain: "Plain text (.txt)",
  json: "JSON",
};

interface InspectorProps {
  phase: Phase;
  result: RunResult | null;
  settings: AppSettings;
  overrides: RunOverrides;
  setOverrides: (next: RunOverrides) => void;
  audioPath: string | null;
  editedSummary: string;
  onRetry: () => void;
}

export function Inspector(props: InspectorProps) {
  const showResult = props.phase === "done" && props.result && props.result.status !== "failed";
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-3 overflow-y-auto border-l border-border bg-panel px-3.5 py-4 scrollbar-thin">
      {showResult ? <ResultInspector {...props} /> : <RunInspector {...props} />}
    </aside>
  );
}

function InfoBox({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <div className="flex flex-col gap-2 rounded-card border border-border bg-panel-soft p-3">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-3 text-base">
          <span className="text-ink-muted">{label}</span>
          <span className="font-semibold text-ink">{value}</span>
        </div>
      ))}
    </div>
  );
}

function RunInspector({ settings, overrides, setOverrides, phase }: InspectorProps) {
  const disabled = phase === "running";
  const patch = (partial: Partial<RunOverrides>) => setOverrides({ ...overrides, ...partial });
  const formats = ["telegram", "json"].map((f) => (f === "telegram" ? "Telegram" : "JSON")).join(", ");

  return (
    <>
      <h2 className="text-[15px] font-semibold text-ink">Настройки запуска</h2>
      <Field label="Провайдер">
        <Select
          disabled={disabled}
          value={overrides.provider}
          options={PROVIDERS}
          onChange={(e) => patch({ provider: e.target.value })}
        />
      </Field>
      <Field label="Модель">
        <Input disabled={disabled} value={overrides.model} onChange={(e) => patch({ model: e.target.value })} />
      </Field>
      <Field label="Язык распознавания">
        <Select
          disabled={disabled}
          value={overrides.transcription_language}
          options={TRANSCRIPTION_LANGS}
          onChange={(e) => patch({ transcription_language: e.target.value })}
        />
      </Field>
      <Field label="Режим резюме">
        <Segmented<SummaryMode>
          value={overrides.mode as SummaryMode}
          onChange={(mode) => patch({ mode })}
          options={[
            { value: "brief", label: "brief" },
            { value: "medium", label: "medium" },
            { value: "detailed", label: "detailed" },
          ]}
        />
      </Field>
      <InfoBox
        rows={[
          ["Chunking", settings.summarization.chunking_mode],
          ["Предобработка", settings.preprocessing.enabled ? "вкл" : "выкл"],
          ["Форматы", formats],
        ]}
      />
      <div className="rounded-card border border-border bg-panel-soft p-3 text-base text-ink-muted">
        Ключи API настраиваются только на экране настроек.
      </div>
    </>
  );
}

function ResultInspector({ result, audioPath, editedSummary, overrides, onRetry }: InspectorProps) {
  const { toast } = useToast();
  const [formats, setFormats] = useState<ExportFormat[]>(["telegram", "plain", "json"]);
  const [busy, setBusy] = useState(false);
  if (!result) return null;

  const partial = result.status === "partial_success";

  const toggle = (f: ExportFormat) =>
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));

  const reveal = async () => {
    const path = result.summary_path ?? result.transcript_path;
    if (!path) return;
    const bridge = await getBridge();
    await bridge.revealPath(path);
  };

  const copy = async () => {
    await navigator.clipboard.writeText(editedSummary);
    toast("Резюме скопировано", "ok");
  };

  const doExport = async () => {
    if (formats.length === 0) {
      toast("Выберите хотя бы один формат", "error");
      return;
    }
    setBusy(true);
    try {
      const basePath = result.summary_path ?? result.transcript_path ?? audioPath ?? "summary";
      const bridge = await getBridge();
      await bridge.exportSummary({
        summary_text: editedSummary,
        formats,
        target_dir: dirName(basePath),
        base_name: stem(audioPath ?? basePath),
        mode: overrides.mode,
      });
      toast("Экспорт завершён", "ok");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Ошибка экспорта", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h2 className="text-[15px] font-semibold text-ink">Результат</h2>
      <InfoBox
        rows={[
          ["Транскрипт", <span key="t" className="text-ok">{result.transcript_path ? "сохранён" : "—"}</span>],
          [
            "Резюме",
            <span key="s" className={partial ? "text-warn" : "text-ok"}>
              {result.summary_path ? "сохранено" : "не создано"}
            </span>,
          ],
          ["Провайдер", overrides.provider],
          ["Модель", overrides.model],
        ]}
      />

      {partial ? (
        <Button variant="primary" size="lg" className="w-full" onClick={onRetry}>
          <RotateCcw className="h-4 w-4" /> Повторить суммаризацию
        </Button>
      ) : (
        <Button variant="primary" size="lg" className="w-full" onClick={copy} disabled={!editedSummary}>
          <Copy className="h-4 w-4" /> Копировать резюме
        </Button>
      )}
      <Button variant="secondary" size="lg" className="w-full" onClick={reveal}>
        <FolderOpen className="h-4 w-4" /> Открыть папку
      </Button>

      <div className="flex flex-col gap-2 rounded-card border border-border bg-panel-soft p-3">
        <h3 className="text-base font-semibold text-ink">Экспорт резюме</h3>
        {(Object.keys(EXPORT_LABELS) as ExportFormat[]).map((f) => (
          <label key={f} className="flex cursor-pointer items-center gap-2 text-base text-ink">
            <input
              type="checkbox"
              className="h-3.5 w-3.5 accent-accent"
              checked={formats.includes(f)}
              onChange={() => toggle(f)}
            />
            {EXPORT_LABELS[f]}
          </label>
        ))}
        <Button variant="secondary" size="lg" className="w-full" onClick={doExport} disabled={busy || !editedSummary}>
          <Download className="h-4 w-4" /> Экспортировать
        </Button>
      </div>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      {children}
    </div>
  );
}
