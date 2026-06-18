import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { getMemoryProviderConfig, saveMemoryProviderConfig } from '@/hermes'
import { Check, Loader2, Save } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { LoadingState, Pill } from './primitives'

/** A field is active only when every clause in its `when` matches the current
 *  values (e.g. Hindsight's `api_url` shows only `when` `mode === 'cloud'`). */
function whenMatches(field: MemoryProviderField, values: Record<string, string>): boolean {
  return field.when.every(clause => String(values[clause.key] ?? '') === clause.value)
}

/** Seed editable values from the schema. Secrets always start blank (their
 *  value is never returned). Conditional fields are seeded against the
 *  unconditional ones first so duplicate keys (same key, different `when`)
 *  resolve to the variant that matches the current selection. */
function seedValues(config: MemoryProviderConfig): Record<string, string> {
  const values: Record<string, string> = {}
  const seed = (field: MemoryProviderField) => {
    values[field.key] = field.kind === 'secret' ? '' : field.value
  }

  config.fields.filter(f => f.when.length === 0).forEach(seed)
  config.fields.filter(f => f.when.length > 0 && whenMatches(f, values)).forEach(seed)
  // Backfill any key not yet seeded (hidden variants) so saves never send undefined.
  config.fields.filter(f => !(f.key in values)).forEach(seed)

  return values
}

function FieldControl({
  field,
  value,
  onChange
}: {
  field: MemoryProviderField
  value: string
  onChange: (value: string) => void
}) {
  if (field.kind === 'select') {
    const selected = field.options.find(option => option.value === value)

    return (
      <>
        <Select onValueChange={onChange} value={value}>
          <SelectTrigger className={CONTROL_TEXT}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {field.options.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {(selected?.description || field.description) && (
          <span className="text-xs text-muted-foreground">{selected?.description || field.description}</span>
        )}
      </>
    )
  }

  if (field.kind === 'boolean') {
    return (
      <div className="flex items-center gap-2">
        <Switch checked={value === 'true'} onCheckedChange={checked => onChange(checked ? 'true' : 'false')} />
        <span className="text-xs text-muted-foreground">{value === 'true' ? 'On' : 'Off'}</span>
      </div>
    )
  }

  if (field.kind === 'secret') {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="min-w-64 flex-1 font-mono"
          onChange={event => onChange(event.target.value)}
          placeholder={field.is_set ? 'Leave blank to keep current value' : field.placeholder}
          type="password"
          value={value}
        />
        {field.is_set && (
          <Pill tone="primary">
            <Check className="size-3" />
            Set
          </Pill>
        )}
      </div>
    )
  }

  return (
    <Input
      className="font-mono"
      onChange={event => onChange(event.target.value)}
      placeholder={field.placeholder}
      value={value}
    />
  )
}

export function ProviderConfigPanel({ provider }: { provider: string }) {
  const [config, setConfig] = useState<MemoryProviderConfig | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getMemoryProviderConfig(provider)
      setConfig(next)
      setValues(seedValues(next))
    } catch (err) {
      notifyError(err, 'Memory provider settings failed to load')
      setConfig(null)
    }
  }, [provider])

  useEffect(() => {
    setConfig(null)
    void refresh()
  }, [refresh])

  const save = useCallback(async () => {
    if (!config) {
      return
    }

    setSaving(true)

    try {
      await saveMemoryProviderConfig(provider, values)
      notify({ kind: 'success', title: `${config.label} saved`, message: 'Memory provider configuration updated.' })
      await refresh()
    } catch (err) {
      notifyError(err, `Failed to save ${config.label} settings`)
    } finally {
      setSaving(false)
    }
  }, [config, provider, refresh, values])

  // Providers without a declared config surface (e.g. builtin) render nothing.
  if (config && config.fields.length === 0) {
    return null
  }

  if (!config) {
    return <LoadingState label="Loading memory provider settings..." />
  }

  const secretFields = config.fields.filter(field => field.kind === 'secret')
  const visibleFields = config.fields.filter(field => whenMatches(field, values))

  return (
    <section className="py-3">
      <button
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 rounded-lg bg-background/60 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setExpanded(open => !open)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <DisclosureCaret open={expanded} />
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {config.label} settings
          </span>
          {secretFields.map(field => (
            <Pill key={field.key}>{field.is_set ? `${field.label} set` : `${field.label} not set`}</Pill>
          ))}
        </span>
      </button>

      {expanded && (
        <div className="mt-3 grid gap-4 rounded-xl bg-background/60 p-4">
          {visibleFields.map((field, index) => (
            <label className="grid gap-1.5" key={`${field.key}-${index}`}>
              <span className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                {field.label}
                {field.required && <span className="text-destructive">*</span>}
                {field.url && (
                  <a
                    className="font-normal text-primary underline-offset-2 hover:underline"
                    href={field.url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    Get key
                  </a>
                )}
              </span>
              <FieldControl
                field={field}
                onChange={value => setValues(current => ({ ...current, [field.key]: value }))}
                value={values[field.key] ?? ''}
              />
              {field.kind !== 'select' && field.description && (
                <span className="text-xs text-muted-foreground">{field.description}</span>
              )}
            </label>
          ))}

          <div className="flex justify-end">
            <Button disabled={saving} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
              Save
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
