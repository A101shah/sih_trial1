/**
 * SatQuery AI Frontend Application Logic
 * Interactive Multimodal Remote-Sensing Assistant Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sampleSelect = document.getElementById("sample-select");
  const dropZone1 = document.getElementById("drop-zone-1");
  const dropZone2 = document.getElementById("drop-zone-2");
  const fileInput1 = document.getElementById("file-input-1");
  const fileInput2 = document.getElementById("file-input-2");
  const previewArea1 = document.getElementById("preview-area-1");
  const previewArea2 = document.getElementById("preview-area-2");
  const imgPreview1 = document.getElementById("img-preview-1");
  const imgPreview2 = document.getElementById("img-preview-2");

  const taskSelect = document.getElementById("task-select");
  const userPrompt = document.getElementById("user-prompt");
  const btnAnalyze = document.getElementById("btn-analyze");
  const analyzeSpinner = document.getElementById("analyze-spinner");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const opacitySlider = document.getElementById("overlay-opacity");

  const viewBaseImg = document.getElementById("view-base-img");
  const viewOverlayImg = document.getElementById("view-overlay-img");
  const viewSplitT1 = document.getElementById("view-split-t1");
  const viewSplitT2 = document.getElementById("view-split-t2");
  const viewMaskImg = document.getElementById("view-mask-img");
  const viewDiffImg = document.getElementById("view-diff-img");
  const bboxLayer = document.getElementById("bbox-overlay-layer");

  const statChangePct = document.getElementById("stat-change-pct");
  const statClusters = document.getElementById("stat-clusters");
  const statConfidence = document.getElementById("stat-confidence");
  const statModel = document.getElementById("stat-model");
  const geoBadge = document.getElementById("geo-badge");

  const aiAnswer = document.getElementById("ai-answer");
  const semanticTags = document.getElementById("semantic-tags");
  const traceList = document.getElementById("trace-list");
  const btnDownloadMd = document.getElementById("btn-download-md");
  const btnDownloadGeoJson = document.getElementById("btn-download-geojson");

  // State
  let image1Data = null;
  let image2Data = null;
  let image1Path = null;
  let image2Path = null;
  let currentResult = null;

  // Initialize: Load sample list from backend
  fetchSamples();

  // Tab switching
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const targetTab = document.getElementById(`tab-${btn.dataset.tab}`);
      if (targetTab) targetTab.classList.add("active");
    });
  });

  // Opacity Slider
  opacitySlider.addEventListener("input", (e) => {
    viewOverlayImg.style.opacity = (e.target.value / 100).toString();
  });

  // Quick Chips
  document.querySelectorAll(".chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      userPrompt.value = chip.dataset.query;
    });
  });

  // Upload Box 1 Click & Drag
  dropZone1.addEventListener("click", () => fileInput1.click());
  setupDragAndDrop(dropZone1, (file) => handleFileUpload(file, 1));
  fileInput1.addEventListener("change", (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0], 1);
  });

  // Upload Box 2 Click & Drag
  dropZone2.addEventListener("click", () => fileInput2.click());
  setupDragAndDrop(dropZone2, (file) => handleFileUpload(file, 2));
  fileInput2.addEventListener("change", (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0], 2);
  });

  // Sample Selection
  sampleSelect.addEventListener("change", (e) => {
    const selected = e.target.value;
    if (!selected) return;
    const sampleData = JSON.parse(selected);
    image1Path = sampleData.t1_path;
    image2Path = sampleData.t2_path;
    image1Data = null;
    image2Data = null;

    // Load preview images
    imgPreview1.src = `/api/v1/sample_image?path=${encodeURIComponent(image1Path)}`;
    imgPreview1.style.display = "block";
    previewArea1.querySelector(".upload-placeholder").style.display = "none";

    imgPreview2.src = `/api/v1/sample_image?path=${encodeURIComponent(image2Path)}`;
    imgPreview2.style.display = "block";
    previewArea2.querySelector(".upload-placeholder").style.display = "none";

    // Set viewport images
    viewBaseImg.src = imgPreview2.src;
    viewSplitT1.src = imgPreview1.src;
    viewSplitT2.src = imgPreview2.src;

    // Reset results view
    viewOverlayImg.style.display = "none";
    if (viewMaskImg) viewMaskImg.src = "";
    if (viewDiffImg) viewDiffImg.src = "";
    if (bboxLayer) bboxLayer.innerHTML = "";
  });

  // Analyze Button
  btnAnalyze.addEventListener("click", executeAnalysis);

  // Download Markdown Report
  btnDownloadMd.addEventListener("click", () => {
    if (!currentResult || !currentResult.markdown_report) return;
    const blob = new Blob([currentResult.markdown_report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `SatQuery_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Download GeoJSON Features
  if (btnDownloadGeoJson) {
    btnDownloadGeoJson.addEventListener("click", () => {
      if (!currentResult || !currentResult.geojson) return;
      const jsonStr = JSON.stringify(currentResult.geojson, null, 2);
      const blob = new Blob([jsonStr], { type: "application/geo+json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `SatQuery_Changes_${Date.now()}.geojson`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function setupDragAndDrop(zone, onDropCallback) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        onDropCallback(e.dataTransfer.files[0]);
      }
    });
  }

  function handleFileUpload(file, targetIndex) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      if (targetIndex === 1) {
        image1Data = dataUrl;
        image1Path = null;
        imgPreview1.src = dataUrl;
        imgPreview1.style.display = "block";
        previewArea1.querySelector(".upload-placeholder").style.display = "none";
        viewBaseImg.src = dataUrl;
        viewSplitT1.src = dataUrl;
      } else {
        image2Data = dataUrl;
        image2Path = null;
        imgPreview2.src = dataUrl;
        imgPreview2.style.display = "block";
        previewArea2.querySelector(".upload-placeholder").style.display = "none";
        viewSplitT2.src = dataUrl;
      }
    };
    reader.readAsDataURL(file);
  }

  async function fetchSamples() {
    try {
      const res = await fetch("/api/v1/samples");
      const data = await res.json();
      sampleSelect.innerHTML = '<option value="">Select pre-loaded sample pair...</option>';

      if (data.custom_samples && data.custom_samples.length > 0) {
        const optGroup = document.createElement("optgroup");
        optGroup.label = "Custom Input Pairs";
        data.custom_samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(s);
          opt.textContent = `${s.name}`;
          optGroup.appendChild(opt);
        });
        sampleSelect.appendChild(optGroup);
      }

      if (data.levir_samples && data.levir_samples.length > 0) {
        const optGroup = document.createElement("optgroup");
        optGroup.label = "LEVIR-CD Benchmark Samples";
        data.levir_samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(s);
          opt.textContent = `LEVIR: ${s.name}`;
          optGroup.appendChild(opt);
        });
        sampleSelect.appendChild(optGroup);
      }
    } catch (err) {
      console.warn("Could not fetch pre-loaded samples:", err);
    }
  }

  async function executeAnalysis() {
    if (!image1Data && !image1Path) {
      alert("Please upload or select at least one primary image.");
      return;
    }

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "block";
    aiAnswer.textContent = "Agent controller analyzing inputs and orchestrating specialist models...";

    const payload = {
      image_1: image1Data,
      image_2: image2Data,
      image_1_path: image1Path,
      image_2_path: image2Path,
      question: userPrompt.value.trim() || undefined,
      task: taskSelect.value === "auto" ? undefined : taskSelect.value
    };

    try {
      const response = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.detail || errJson.error || "Analysis failed");
      }

      const result = await response.json();
      currentResult = result;
      updateUIWithResults(result);

    } catch (err) {
      aiAnswer.innerHTML = `<span style="color: var(--accent-red)">Error: ${err.message}</span>`;
      console.error(err);
    } finally {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = "none";
    }
  }

  function updateUIWithResults(result) {
    // 1. AI Answer
    aiAnswer.textContent = result.answer || "Analysis complete.";

    // 2. Metrics Bar
    if (result.change_percentage !== undefined && result.task === "change_analysis") {
      statChangePct.textContent = `${result.change_percentage.toFixed(2)} %`;
      statClusters.textContent = result.bboxes ? result.bboxes.length : "0";
    } else if (result.count !== undefined) {
      statChangePct.textContent = "N/A";
      statClusters.textContent = `Count: ${result.count}`;
    } else if (result.num_regions !== undefined) {
      statChangePct.textContent = "N/A";
      statClusters.textContent = `Regions: ${result.num_regions}`;
    } else {
      statChangePct.textContent = "N/A";
      statClusters.textContent = "N/A";
    }

    // Confidence
    if (result.confidence !== null && result.confidence !== undefined) {
      statConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
      statConfidence.title = result.confidence_source || "Model Probability";
    } else {
      statConfidence.textContent = "N/A";
      statConfidence.title = "Uncertainty unavailable";
    }

    // Specialist
    if (result.task === "change_analysis") {
      statModel.textContent = "ChangeFormerV6";
    } else if (result.models_used && result.models_used.length > 0) {
      statModel.textContent = result.models_used[0].split(" ")[0];
    } else {
      statModel.textContent = result.task.toUpperCase();
    }

    // Geospatial Badge
    if (result.geospatial_metadata && result.geospatial_metadata.t1 && result.geospatial_metadata.t1.crs) {
      geoBadge.textContent = `CRS: ${result.geospatial_metadata.t1.crs.split(" ")[0]}`;
      geoBadge.style.display = "inline-block";
    } else {
      geoBadge.textContent = "Geo: Standard";
    }

    // 3. Overlays & Visualizations
    if (result.grounding_vis_url) {
      viewOverlayImg.src = result.grounding_vis_url;
      viewOverlayImg.style.display = "block";
      viewBaseImg.src = result.grounding_vis_url;
    } else if (result.fusion_vis_url) {
      viewOverlayImg.src = result.fusion_vis_url;
      viewOverlayImg.style.display = "block";
      viewBaseImg.src = result.fusion_vis_url;
    } else if (result.overlay_t2_url) {
      viewOverlayImg.src = result.overlay_t2_url;
      viewOverlayImg.style.display = "block";
    } else {
      viewOverlayImg.style.display = "none";
    }

    if (result.change_mask_url && viewMaskImg) {
      viewMaskImg.src = result.change_mask_url;
    }
    if (result.difference_map_url && viewDiffImg) {
      viewDiffImg.src = result.difference_map_url;
    }

    // 4. Specialist Status Tags
    semanticTags.innerHTML = "";
    if (result.top_predictions && result.top_predictions.length > 0) {
      result.top_predictions.forEach(p => {
        const tag = document.createElement("span");
        tag.className = "tag tag-executed";
        tag.textContent = `${p.class} (${(p.probability * 100).toFixed(0)}%)`;
        semanticTags.appendChild(tag);
      });
    } else if (result.specialist_statuses) {
      for (const [spec, status] of Object.entries(result.specialist_statuses)) {
        const tag = document.createElement("span");
        tag.className = status === "EXECUTED" ? "tag tag-executed" : "tag tag-unconfigured";
        tag.textContent = `${spec}: ${status}`;
        semanticTags.appendChild(tag);
      }
    }

    // 5. Execution Trace Timeline
    traceList.innerHTML = "";
    if (result.trace) {
      result.trace.forEach((step, idx) => {
        const li = document.createElement("li");
        li.className = "trace-item complete";
        li.innerHTML = `<span class="step-num">${idx + 1}</span> ${step}`;
        traceList.appendChild(li);
      });
    }

    // 6. Report Export Actions
    btnDownloadMd.disabled = false;
    if (btnDownloadGeoJson) {
      btnDownloadGeoJson.disabled = !result.geojson;
    }
  }
});
