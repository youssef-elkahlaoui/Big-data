/**
 * Performance Optimized JavaScript for Food Recommender App
 * Progressive Enhancement with lazy loading and smooth interactions
 */

// Performance utilities
const perfUtils = {
  // Debounce function for search inputs
  debounce: (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // Throttle function for scroll events
  throttle: (func, limit) => {
    let inThrottle;
    return function () {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  },

  // Lazy load images
  lazyLoadImages: () => {
    const images = document.querySelectorAll("img[data-src]");
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.remove("lazy");
          imageObserver.unobserve(img);
        }
      });
    });
    images.forEach((img) => imageObserver.observe(img));
  },

  // Preload critical resources
  preloadResource: (href, as, type = null) => {
    const link = document.createElement("link");
    link.rel = "preload";
    link.href = href;
    link.as = as;
    if (type) link.type = type;
    document.head.appendChild(link);
  },
};

// Enhanced search functionality with autocomplete
class SearchEnhancer {
  constructor() {
    this.searchInput = document.querySelector(
      '#search-input, input[name="query"]'
    );
    this.searchForm = document.querySelector('form[action*="search"]');
    this.resultsContainer = document.querySelector("#search-results");
    this.init();
  }

  init() {
    if (!this.searchInput) return;

    // Add search suggestions
    this.setupAutocomplete();

    // Add loading states
    this.setupLoadingStates();

    // Add keyboard navigation
    this.setupKeyboardNavigation();
  }

  setupAutocomplete() {
    const searchSuggestions = perfUtils.debounce(async (query) => {
      if (query.length < 2) return;

      try {
        const response = await fetch(
          `/api/search_suggestions?q=${encodeURIComponent(query)}`
        );
        if (response.ok) {
          const suggestions = await response.json();
          this.showSuggestions(suggestions);
        }
      } catch (error) {
        console.warn("Search suggestions failed:", error);
      }
    }, 300);

    this.searchInput.addEventListener("input", (e) => {
      searchSuggestions(e.target.value);
    });
  }

  setupLoadingStates() {
    if (!this.searchForm) return;

    this.searchForm.addEventListener("submit", (e) => {
      const submitBtn = this.searchForm.querySelector('button[type="submit"]');
      if (submitBtn) {
        submitBtn.innerHTML =
          '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';
        submitBtn.disabled = true;
      }
    });
  }

  setupKeyboardNavigation() {
    this.searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.hideSuggestions();
      }
    });
  }

  showSuggestions(suggestions) {
    // Implementation for showing autocomplete suggestions
    // This would create a dropdown with suggestions
  }

  hideSuggestions() {
    const suggestionsBox = document.querySelector(".search-suggestions");
    if (suggestionsBox) {
      suggestionsBox.style.display = "none";
    }
  }
}

// Progressive Web App features
class PWAEnhancer {
  constructor() {
    this.init();
  }

  init() {
    // Service Worker registration
    this.registerServiceWorker();

    // Install prompt
    this.setupInstallPrompt();

    // Offline detection
    this.setupOfflineDetection();
  }

  registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .then((registration) => console.log("SW registered"))
        .catch((error) => console.log("SW registration failed"));
    }
  }

  setupInstallPrompt() {
    let deferredPrompt;

    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      deferredPrompt = e;

      // Show install button
      const installBtn = document.querySelector("#install-app");
      if (installBtn) {
        installBtn.style.display = "block";
        installBtn.addEventListener("click", () => {
          deferredPrompt.prompt();
          deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === "accepted") {
              console.log("User accepted the install prompt");
            }
            deferredPrompt = null;
          });
        });
      }
    });
  }

  setupOfflineDetection() {
    window.addEventListener("online", () => {
      this.showConnectionStatus("Connected", "success");
    });

    window.addEventListener("offline", () => {
      this.showConnectionStatus("Offline Mode", "warning");
    });
  }

  showConnectionStatus(message, type) {
    const toast = document.createElement("div");
    toast.className = `fixed top-4 right-4 px-4 py-2 rounded-lg ${
      type === "success" ? "bg-green-500" : "bg-yellow-500"
    } text-white z-50`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 3000);
  }
}

// Performance monitoring
class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.init();
  }

  init() {
    // Core Web Vitals
    this.measureCoreWebVitals();

    // Custom metrics
    this.measurePageLoadTime();

    // Report metrics
    this.reportMetrics();
  }

  measureCoreWebVitals() {
    // Largest Contentful Paint
    new PerformanceObserver((entryList) => {
      const entries = entryList.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.metrics.lcp = lastEntry.startTime;
    }).observe({ entryTypes: ["largest-contentful-paint"] });

    // First Input Delay
    new PerformanceObserver((entryList) => {
      const firstInput = entryList.getEntries()[0];
      this.metrics.fid = firstInput.processingStart - firstInput.startTime;
    }).observe({ entryTypes: ["first-input"] });

    // Cumulative Layout Shift
    new PerformanceObserver((entryList) => {
      let clsValue = 0;
      for (const entry of entryList.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      this.metrics.cls = clsValue;
    }).observe({ entryTypes: ["layout-shift"] });
  }

  measurePageLoadTime() {
    window.addEventListener("load", () => {
      const loadTime =
        performance.timing.loadEventEnd - performance.timing.navigationStart;
      this.metrics.pageLoadTime = loadTime;
    });
  }

  reportMetrics() {
    // Report to analytics or monitoring service
    setTimeout(() => {
      console.log("Performance Metrics:", this.metrics);
      // Could send to Google Analytics, monitoring service, etc.
    }, 5000);
  }
}

// Initialize everything when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  // Basic performance optimizations
  perfUtils.lazyLoadImages();

  // Enhanced features
  new SearchEnhancer();
  new PWAEnhancer();
  new PerformanceMonitor();

  // Mobile menu functionality
  const mobileMenuBtn = document.querySelector("#mobile-menu-button");
  const mobileMenu = document.querySelector("#mobile-menu");

  if (mobileMenuBtn && mobileMenu) {
    mobileMenuBtn.addEventListener("click", () => {
      mobileMenu.classList.toggle("hidden");
    });
  }

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  // Form enhancements
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", function (e) {
      const submitBtn = this.querySelector('button[type="submit"]');
      if (submitBtn && !submitBtn.disabled) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML =
          '<i class="fas fa-spinner fa-spin mr-2"></i>Loading...';
        submitBtn.disabled = true;

        // Re-enable after 10 seconds as fallback
        setTimeout(() => {
          submitBtn.innerHTML = originalText;
          submitBtn.disabled = false;
        }, 10000);
      }
    });
  });

  // Auto-hide flash messages
  setTimeout(() => {
    document.querySelectorAll(".alert").forEach((alert) => {
      alert.style.transition = "opacity 0.5s ease-out";
      alert.style.opacity = "0";
      setTimeout(() => alert.remove(), 500);
    });
  }, 5000);
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    perfUtils,
    SearchEnhancer,
    PWAEnhancer,
    PerformanceMonitor,
  };
}
