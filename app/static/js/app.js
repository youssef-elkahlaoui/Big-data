// Main JavaScript for Food Recommender System
// Global variables and utilities

let currentUser = null;
let searchHistory = [];
let comparisonList = [];
let favorites = [];

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
  loadUserPreferences();
  setupEventListeners();
});

// Initialize application components
function initializeApp() {
  // Initialize tooltips
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize popovers
  var popoverTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="popover"]')
  );
  var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
  });

  // Load saved data from localStorage
  loadFromStorage();

  // Setup search suggestions
  setupSearchSuggestions();

  // Update UI with loaded data
  updateUI();
}

// Setup global event listeners
function setupEventListeners() {
  // Search form enhancements
  const searchForms = document.querySelectorAll('form[action*="search"]');
  searchForms.forEach((form) => {
    form.addEventListener("submit", function (e) {
      const query = form.querySelector('input[name="query"]').value.trim();
      if (query) {
        addToSearchHistory(query);
      }
    });
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    // Ctrl/Cmd + K for quick search
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      focusSearch();
    }

    // Escape to close modals
    if (e.key === "Escape") {
      closeModals();
    }
  });

  // Scroll to top button
  const scrollTopBtn = document.getElementById("scrollTopBtn");
  if (scrollTopBtn) {
    window.addEventListener("scroll", function () {
      if (window.pageYOffset > 300) {
        scrollTopBtn.style.display = "block";
      } else {
        scrollTopBtn.style.display = "none";
      }
    });

    scrollTopBtn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
}

// Search functionality
function setupSearchSuggestions() {
  const searchInputs = document.querySelectorAll('input[name="query"]');

  searchInputs.forEach((input) => {
    let suggestionTimeout;

    input.addEventListener("input", function () {
      clearTimeout(suggestionTimeout);
      const query = this.value.trim();

      if (query.length < 2) {
        hideSuggestions();
        return;
      }

      suggestionTimeout = setTimeout(() => {
        fetchSuggestions(query, this);
      }, 300);
    });

    // Hide suggestions when clicking outside
    document.addEventListener("click", function (e) {
      if (!input.contains(e.target)) {
        hideSuggestions();
      }
    });
  });
}

async function fetchSuggestions(query, inputElement) {
  try {
    const response = await fetch(
      `/api/suggestions?q=${encodeURIComponent(query)}&limit=8`
    );
    const data = await response.json();

    if (data.success) {
      showSuggestions(data.suggestions, inputElement);
    }
  } catch (error) {
    console.error("Error fetching suggestions:", error);
  }
}

function showSuggestions(suggestions, inputElement) {
  hideSuggestions(); // Remove any existing suggestions

  if (suggestions.length === 0) return;

  const suggestionsList = document.createElement("div");
  suggestionsList.className = "search-suggestions";
  suggestionsList.innerHTML = suggestions
    .map(
      (suggestion) => `
        <div class="suggestion-item" onclick="selectSuggestion('${
          suggestion.text
        }', '${inputElement.name}')">
            <i class="fas fa-search me-2 text-muted"></i>
            ${suggestion.text}
            ${
              suggestion.type
                ? `<span class="suggestion-type">${suggestion.type}</span>`
                : ""
            }
        </div>
    `
    )
    .join("");

  // Position and show suggestions
  const rect = inputElement.getBoundingClientRect();
  suggestionsList.style.position = "absolute";
  suggestionsList.style.top = rect.bottom + window.scrollY + "px";
  suggestionsList.style.left = rect.left + "px";
  suggestionsList.style.width = rect.width + "px";
  suggestionsList.style.zIndex = "1000";

  document.body.appendChild(suggestionsList);
}

function hideSuggestions() {
  const existing = document.querySelector(".search-suggestions");
  if (existing) {
    existing.remove();
  }
}

function selectSuggestion(text, inputName) {
  const input = document.querySelector(`input[name="${inputName}"]`);
  if (input) {
    input.value = text;
    input.form.submit();
  }
  hideSuggestions();
}

// Search history management
function addToSearchHistory(query) {
  if (!searchHistory.includes(query)) {
    searchHistory.unshift(query);
    searchHistory = searchHistory.slice(0, 10); // Keep only last 10 searches
    saveToStorage();
  }
}

function clearSearchHistory() {
  searchHistory = [];
  saveToStorage();
  updateUI();
}

// Comparison functionality
function addToComparison(productCode, productName) {
  if (comparisonList.find((p) => p.code === productCode)) {
    showNotification("Product already in comparison list", "warning");
    return;
  }

  if (comparisonList.length >= 4) {
    showNotification("Maximum 4 products can be compared at once", "warning");
    return;
  }

  comparisonList.push({ code: productCode, name: productName });
  saveToStorage();
  updateComparisonUI();
  showNotification(`${productName} added to comparison`, "success");
}

function removeFromComparison(productCode) {
  comparisonList = comparisonList.filter((p) => p.code !== productCode);
  saveToStorage();
  updateComparisonUI();
}

function clearComparison() {
  comparisonList = [];
  saveToStorage();
  updateComparisonUI();
}

function compareProducts() {
  if (comparisonList.length < 2) {
    showNotification("Select at least 2 products to compare", "warning");
    return;
  }

  const productCodes = comparisonList.map((p) => p.code).join(",");
  window.location.href = `/nutrition-comparison?products=${productCodes}`;
}

// Favorites functionality
function toggleFavorite(productCode, productName) {
  const existingIndex = favorites.findIndex((f) => f.code === productCode);

  if (existingIndex !== -1) {
    favorites.splice(existingIndex, 1);
    showNotification(`${productName} removed from favorites`, "info");
  } else {
    favorites.push({
      code: productCode,
      name: productName,
      addedAt: new Date().toISOString(),
    });
    showNotification(`${productName} added to favorites`, "success");
  }

  saveToStorage();
  updateFavoritesUI();
}

function clearFavorites() {
  favorites = [];
  saveToStorage();
  updateFavoritesUI();
}

// UI Update functions
function updateUI() {
  updateComparisonUI();
  updateFavoritesUI();
  updateSearchHistoryUI();
}

function updateComparisonUI() {
  // Update comparison badge
  const comparisonBadges = document.querySelectorAll(".comparison-badge");
  comparisonBadges.forEach((badge) => {
    badge.textContent = comparisonList.length;
    badge.style.display = comparisonList.length > 0 ? "inline" : "none";
  });

  // Update comparison dropdown
  const comparisonDropdown = document.getElementById("comparisonDropdown");
  if (comparisonDropdown) {
    if (comparisonList.length === 0) {
      comparisonDropdown.innerHTML =
        '<li class="dropdown-item-text text-muted">No products to compare</li>';
    } else {
      comparisonDropdown.innerHTML =
        comparisonList
          .map(
            (product) => `
                <li class="d-flex justify-content-between align-items-center px-3 py-1">
                    <span class="small">${product.name}</span>
                    <button class="btn btn-sm btn-link text-danger p-0" onclick="removeFromComparison('${product.code}')">
                        <i class="fas fa-times"></i>
                    </button>
                </li>
            `
          )
          .join("") +
        `
                <li><hr class="dropdown-divider"></li>
                <li class="px-3 py-1">
                    <button class="btn btn-primary btn-sm w-100" onclick="compareProducts()">
                        <i class="fas fa-chart-bar me-1"></i>Compare Now
                    </button>
                </li>
            `;
    }
  }
}

function updateFavoritesUI() {
  // Update favorites badge
  const favoritesBadges = document.querySelectorAll(".favorites-badge");
  favoritesBadges.forEach((badge) => {
    badge.textContent = favorites.length;
    badge.style.display = favorites.length > 0 ? "inline" : "none";
  });
}

function updateSearchHistoryUI() {
  const searchHistoryContainer = document.getElementById("searchHistory");
  if (searchHistoryContainer) {
    if (searchHistory.length === 0) {
      searchHistoryContainer.innerHTML =
        '<p class="text-muted">No recent searches</p>';
    } else {
      searchHistoryContainer.innerHTML = searchHistory
        .map(
          (query) => `
                <div class="search-history-item">
                    <a href="/search?query=${encodeURIComponent(
                      query
                    )}" class="text-decoration-none">
                        <i class="fas fa-history me-2 text-muted"></i>${query}
                    </a>
                </div>
            `
        )
        .join("");
    }
  }
}

// Storage functions
function saveToStorage() {
  try {
    localStorage.setItem(
      "foodRecommender_searchHistory",
      JSON.stringify(searchHistory)
    );
    localStorage.setItem(
      "foodRecommender_comparisonList",
      JSON.stringify(comparisonList)
    );
    localStorage.setItem(
      "foodRecommender_favorites",
      JSON.stringify(favorites)
    );
  } catch (error) {
    console.error("Error saving to localStorage:", error);
  }
}

function loadFromStorage() {
  try {
    const savedSearchHistory = localStorage.getItem(
      "foodRecommender_searchHistory"
    );
    if (savedSearchHistory) {
      searchHistory = JSON.parse(savedSearchHistory);
    }

    const savedComparisonList = localStorage.getItem(
      "foodRecommender_comparisonList"
    );
    if (savedComparisonList) {
      comparisonList = JSON.parse(savedComparisonList);
    }

    const savedFavorites = localStorage.getItem("foodRecommender_favorites");
    if (savedFavorites) {
      favorites = JSON.parse(savedFavorites);
    }
  } catch (error) {
    console.error("Error loading from localStorage:", error);
  }
}

function loadUserPreferences() {
  try {
    const preferences = localStorage.getItem("foodRecommender_preferences");
    if (preferences) {
      const prefs = JSON.parse(preferences);
      applyUserPreferences(prefs);
    }
  } catch (error) {
    console.error("Error loading user preferences:", error);
  }
}

function applyUserPreferences(prefs) {
  // Apply theme
  if (prefs.theme) {
    document.body.setAttribute("data-theme", prefs.theme);
  }

  // Apply other preferences
  if (prefs.compactView) {
    document.body.classList.add("compact-view");
  }
}

// Utility functions
function focusSearch() {
  const searchInput = document.querySelector('input[name="query"]');
  if (searchInput) {
    searchInput.focus();
    searchInput.select();
  }
}

function closeModals() {
  const modals = document.querySelectorAll(".modal.show");
  modals.forEach((modal) => {
    const modalInstance = bootstrap.Modal.getInstance(modal);
    if (modalInstance) {
      modalInstance.hide();
    }
  });
}

function showNotification(message, type = "info") {
  // Create notification element
  const notification = document.createElement("div");
  notification.className = `alert alert-${type} notification-toast`;
  notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close ms-auto" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;

  // Add to notifications container
  let container = document.getElementById("notificationsContainer");
  if (!container) {
    container = document.createElement("div");
    container.id = "notificationsContainer";
    container.className = "notifications-container";
    document.body.appendChild(container);
  }

  container.appendChild(notification);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 5000);
}

function getNotificationIcon(type) {
  const icons = {
    success: "check-circle",
    warning: "exclamation-triangle",
    danger: "times-circle",
    info: "info-circle",
  };
  return icons[type] || "info-circle";
}

// API helper functions
async function apiRequest(url, options = {}) {
  try {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("API request failed:", error);
    throw error;
  }
}

// Product card interactions
function handleProductCardClick(event, productCode) {
  // Don't navigate if clicking on action buttons
  if (event.target.closest(".btn, .product-actions")) {
    return;
  }

  window.location.href = `/product/${productCode}`;
}

// Loading states
function showLoading(element, message = "Loading...") {
  element.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">${message}</span>
            </div>
            <p class="mt-2 text-muted">${message}</p>
        </div>
    `;
}

function hideLoading(element) {
  const spinner = element.querySelector(".spinner-border");
  if (spinner) {
    spinner.parentElement.remove();
  }
}

// Form validation helpers
function validateEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

function validateRequired(value) {
  return value && value.trim().length > 0;
}

// Export functions for global use
window.FoodRecommender = {
  addToComparison,
  removeFromComparison,
  clearComparison,
  compareProducts,
  toggleFavorite,
  clearFavorites,
  showNotification,
  apiRequest,
  handleProductCardClick,
};

// Analytics tracking (if needed)
function trackEvent(category, action, label = null, value = null) {
  if (typeof gtag !== "undefined") {
    gtag("event", action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
}

// Performance monitoring
function measurePerformance(name, fn) {
  const start = performance.now();
  const result = fn();
  const end = performance.now();
  console.log(`${name} took ${end - start} milliseconds`);
  return result;
}

// Error boundary for uncaught errors
window.addEventListener("error", function (event) {
  console.error("Uncaught error:", event.error);
  showNotification(
    "An unexpected error occurred. Please refresh the page.",
    "danger"
  );
});

window.addEventListener("unhandledrejection", function (event) {
  console.error("Unhandled promise rejection:", event.reason);
  showNotification(
    "A network error occurred. Please check your connection.",
    "warning"
  );
});
