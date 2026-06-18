import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string) => getMemoryProviderConfig(provider),
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function field(partial: Partial<MemoryProviderField> & Pick<MemoryProviderField, 'key'>): MemoryProviderField {
  return {
    default: '',
    description: '',
    is_set: false,
    kind: 'text',
    label: partial.key,
    options: [],
    placeholder: '',
    required: false,
    url: '',
    value: '',
    value_type: 'str',
    when: [],
    ...partial
  }
}

function hindsightSchema(overrides: Partial<MemoryProviderField>[] = []): MemoryProviderConfig {
  const fields: MemoryProviderField[] = [
    field({
      key: 'mode',
      label: 'Mode',
      kind: 'select',
      value: 'cloud',
      is_set: true,
      description: 'How Hermes connects to Hindsight.',
      options: [
        { value: 'cloud', label: 'Cloud', description: 'Hindsight Cloud API (lightweight, just needs an API key)' },
        { value: 'local_external', label: 'Local External', description: 'Connect to an existing Hindsight instance' }
      ]
    }),
    field({
      key: 'api_key',
      label: 'API key',
      kind: 'secret',
      description: 'Used to authenticate with the Hindsight API.',
      placeholder: 'Enter Hindsight API key'
    }),
    field({ key: 'api_url', label: 'API URL', value: 'https://api.hindsight.vectorize.io', is_set: true }),
    field({ key: 'bank_id', label: 'Bank ID', value: 'hermes', is_set: true }),
    field({
      key: 'recall_budget',
      label: 'Recall budget',
      kind: 'select',
      value: 'mid',
      is_set: true,
      options: [
        { value: 'low', label: 'low', description: '' },
        { value: 'mid', label: 'mid', description: '' },
        { value: 'high', label: 'high', description: '' }
      ]
    })
  ]

  return {
    name: 'hindsight',
    label: 'Hindsight',
    fields: fields.map((f, index) => ({ ...f, ...overrides[index] }))
  }
}

beforeEach(() => {
  getMemoryProviderConfig.mockResolvedValue(hindsightSchema())
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel(provider = 'hindsight') {
  const { ProviderConfigPanel } = await import('./provider-config-panel')

  return render(<ProviderConfigPanel provider={provider} />)
}

describe('ProviderConfigPanel', () => {
  it('renders the declared provider fields generically', async () => {
    await renderPanel()

    expect(await screen.findByDisplayValue('https://api.hindsight.vectorize.io')).toBeTruthy()
    expect(screen.getByDisplayValue('hermes')).toBeTruthy()
    expect(screen.getByText('Cloud')).toBeTruthy()
    expect(screen.getAllByText('Hindsight Cloud API (lightweight, just needs an API key)').length).toBeGreaterThan(0)
    expect(screen.getByText('mid')).toBeTruthy()
  })

  it('collapses and expands the fields', async () => {
    await renderPanel()

    expect(await screen.findByLabelText('API URL')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(screen.queryByLabelText('API URL')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(await screen.findByLabelText('API URL')).toBeTruthy()
  })

  it('saves edited values without requiring a secret replacement', async () => {
    await renderPanel()

    const apiUrl = await screen.findByLabelText('API URL')
    fireEvent.change(apiUrl, { target: { value: 'http://localhost:8888' } })
    fireEvent.change(screen.getByLabelText('Bank ID'), { target: { value: 'ben-bank' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('hindsight', {
        mode: 'cloud',
        api_key: '',
        api_url: 'http://localhost:8888',
        bank_id: 'ben-bank',
        recall_budget: 'mid'
      })
    )
  })

  it('renders nothing for a provider with no declared config surface', async () => {
    getMemoryProviderConfig.mockResolvedValue({ name: 'builtin', label: 'builtin', fields: [] })

    const { container } = await renderPanel('builtin')

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('builtin'))
    expect(container.querySelector('section')).toBeNull()
  })

  it('shows and hides fields based on their when-clause as the mode changes', async () => {
    getMemoryProviderConfig.mockResolvedValue({
      name: 'hindsight',
      label: 'Hindsight',
      fields: [
        field({
          key: 'mode',
          label: 'Mode',
          kind: 'select',
          value: 'cloud',
          options: [
            { value: 'cloud', label: 'Cloud', description: '' },
            { value: 'local_embedded', label: 'Local Embedded', description: '' }
          ]
        }),
        field({ key: 'api_url', label: 'API URL', value: 'https://api', when: [{ key: 'mode', value: 'cloud' }] }),
        field({ key: 'llm_model', label: 'LLM model', value: 'gpt-4o-mini', when: [{ key: 'mode', value: 'local_embedded' }] })
      ]
    })

    await renderPanel()

    // Cloud is selected: the cloud-gated field shows, the embedded one doesn't.
    expect(await screen.findByLabelText('API URL')).toBeTruthy()
    expect(screen.queryByLabelText('LLM model')).toBeNull()

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByRole('option', { name: 'Local Embedded' }))

    // Switching to local_embedded flips which gated field is visible.
    expect(await screen.findByLabelText('LLM model')).toBeTruthy()
    expect(screen.queryByLabelText('API URL')).toBeNull()
  })

  it('renders a boolean field as a toggle and saves it as a string', async () => {
    getMemoryProviderConfig.mockResolvedValue({
      name: 'hindsight',
      label: 'Hindsight',
      fields: [field({ key: 'auto_recall', label: 'Auto recall', kind: 'boolean', value: 'true', value_type: 'bool', is_set: true })]
    })

    await renderPanel()

    const toggle = await screen.findByRole('switch')
    expect(toggle).toBeTruthy()
    fireEvent.click(toggle)
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(saveMemoryProviderConfig).toHaveBeenCalledWith('hindsight', { auto_recall: 'false' }))
  })
})
