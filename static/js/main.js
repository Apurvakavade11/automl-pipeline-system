// Show loading spinner
function showLoading() {
    $('.spinner-overlay').addClass('active');
}

// Hide loading spinner
function hideLoading() {
    $('.spinner-overlay').removeClass('active');
}

// Handle file upload with preview
$(document).ready(function() {
    // Auto-detect target column
    $('#target_column').on('change', function() {
        const targetColumn = $(this).val();
        
        $.ajax({
            url: '/detect_target',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({target_column: targetColumn}),
            success: function(response) {
                $('#problem_type').html(`
                    <div class="alert alert-info">
                        <strong>Problem Type Detected:</strong> ${response.problem_type}<br>
                        <strong>Task:</strong> ${response.task}
                        ${response.n_classes ? `<br><strong>Number of Classes:</strong> ${response.n_classes}` : ''}
                    </div>
                `);
                $('#train_button').prop('disabled', false);
            }
        });
    });
    
    // Train model form submission
    $('#train_form').on('submit', function(e) {
        showLoading();
    });
    
    // Split dataset
    $('#split_form').on('submit', function(e) {
        showLoading();
    });
    
    // Predict form submission
    $('#predict_form').on('submit', function(e) {
        showLoading();
    });
    
    // Load insights
    $('#load_insights').on('click', function() {
        $.ajax({
            url: '/get_insights',
            method: 'GET',
            success: function(data) {
                let html = '<h5>Dataset Insights</h5>';
                html += '<p><strong>Shape:</strong> ' + data.shape[0] + ' rows, ' + data.shape[1] + ' columns</p>';
                html += '<h6>Missing Values:</h6><ul>';
                for (let [col, missing] of Object.entries(data.missing_values)) {
                    if (missing > 0) {
                        html += `<li>${col}: ${missing} missing values</li>`;
                    }
                }
                html += '</ul>';
                html += '<h6>Summary Statistics:</h6>';
                html += data.summary_stats;
                $('#insights_content').html(html);
            }
        });
    });
});

// Function to download predictions
function downloadPredictions(filename) {
    window.location.href = '/download/' + filename;
}

// Function to view model results
function viewResults(sessionId) {
    window.location.href = '/results';
}

// Function to copy code snippet
function copyCode(elementId) {
    const code = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(code);
    alert('Code copied to clipboard!');
}