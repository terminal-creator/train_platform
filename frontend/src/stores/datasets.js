import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDatasetsStore = defineStore('datasets', () => {
  // Persist selected dataset UUID across route changes
  const selectedDatasetUuid = ref(null)

  const setSelectedDataset = (uuid) => {
    selectedDatasetUuid.value = uuid
  }

  const clearSelectedDataset = () => {
    selectedDatasetUuid.value = null
  }

  return {
    selectedDatasetUuid,
    setSelectedDataset,
    clearSelectedDataset
  }
})
