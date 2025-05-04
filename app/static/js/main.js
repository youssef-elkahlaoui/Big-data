// Main JavaScript file for the Food Recommender System

document.addEventListener("DOMContentLoaded", function () {
  // Initialize search form submission with loading indicators
  initSearchForms();

  // Initialize tooltips if Bootstrap is available
  if (typeof bootstrap !== "undefined" && bootstrap.Tooltip) {
    const tooltipTriggerList = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }
});

function initSearchForms() {
  // Add loading indicators to search forms
  const searchForms = document.querySelectorAll('form[action*="search"]');

  searchForms.forEach((form) => {
    form.addEventListener("submit", function () {
      const submitBtn = this.querySelector('button[type="submit"]');
      if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="loading me-2"></span> Searching...';
        submitBtn.disabled = true;

        // Re-enable button after 10 seconds (in case of errors)
        setTimeout(() => {
          submitBtn.innerHTML = originalText;
          submitBtn.disabled = false;
        }, 10000);
      }
    });
  });
}

// Function to fetch recommendations via API
function fetchRecommendations(productCode, type = "similar") {
  const resultsContainer = document.getElementById(type + "-recommendations");
  if (!resultsContainer) return;

  resultsContainer.innerHTML =
    '<div class="text-center py-4"><span class="loading"></span> Loading recommendations...</div>';

  fetch(`/api/recommend?product_code=${productCode}&type=${type}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((data) => {
      displayRecommendations(data.products, resultsContainer);
    })
    .catch((error) => {
      resultsContainer.innerHTML = `<div class="alert alert-danger">Failed to load recommendations: ${error.message}</div>`;
    });
}

// Function to display recommendation results
function displayRecommendations(products, container) {
  if (!products || products.length === 0) {
    container.innerHTML =
      '<div class="alert alert-info">No recommendations found.</div>';
    return;
  }

  let html = '<div class="row">';

  products.forEach((product) => {
    const nutriscoreClass = getNutriscoreClass(product.nutriscore_grade);
    const similarity = product.similarity_score
      ? Math.round(product.similarity_score * 100)
      : "";

    html += `
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">${product.product_name}</h5>
                        <p class="card-text">
                            <span class="badge bg-${nutriscoreClass}">
                                Nutriscore ${product.nutriscore_grade.toUpperCase()}
                            </span>
                        </p>
                        ${
                          similarity
                            ? `<p class="card-text"><small>Similarity: ${similarity}%</small></p>`
                            : ""
                        }
                    </div>
                    <div class="card-footer bg-white">
                        <a href="/product/${
                          product.code
                        }" class="btn btn-sm btn-outline-primary">View Details</a>
                    </div>
                </div>
            </div>
        `;
  });

  html += "</div>";
  container.innerHTML = html;
}

// Helper function to get Bootstrap class for a nutriscore
function getNutriscoreClass(nutriscore) {
  if (!nutriscore) return "secondary";

  nutriscore = nutriscore.toLowerCase();

  if (nutriscore === "a" || nutriscore === "b") {
    return "success";
  } else if (nutriscore === "c" || nutriscore === "d") {
    return "warning";
  } else if (nutriscore === "e") {
    return "danger";
  } else {
    return "secondary";
  }
}
