<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>FitLins - {{ level }}-level report</title>
<style type="text/css">
.sub-report-title {}
.summary-heading {}
.run-title {}
.elem-title {}
.elem-desc {}
.elem-filename {}
.warning {
    border: 1px solid #ffaaaa;
    background: #ffe8e8;
    padding: 0.8em;
}
summary.heading-1 {
    font-size: 12pt;
    font-weight: bold;
}
summary.heading-2 {
    font-size: 11pt;
    font-weight: bold;
}
</style>
</head>
<body>
    <div id="summary">
        <h1 class="sub-report-title">Summary</h1>
        <ul class="elem-desc">
            <li>Dataset: {{ dataset.name }}{% if dataset.doi %} (doi:<a href="https://doi.org/{{ dataset.doi }}">{{ dataset.doi }}</a>){% endif %}</li>
            <li>Model: {{ model.name }}</li>
            <li>Participants ({{ subjects|count }}): {{ subjects|join(', ') }}
        </ul>
    </div>
    <div id="model">
        <h1 class="sub-report-title">Model</h1>
        <details>
        <summary>Model specification</summary>
        <pre>{{ model|tojson(indent=2) }}</pre>
        </details>

        <!-- { % for step in steps % } -->
        {% set step = steps.0 %}
        <h2>{{ step.name|capitalize }} level</h2>

        <!-- { % if loop.first %} -->
        <h3>Design matrices</h3>
        <p>A design matrix was generated for each {{ step.name }}. All but the
        first are collapsed, but each should be inspected for correctness.
        {% for analysis in step.analyses %}
        <details{% if loop.first %} open{% endif %}>
        <summary class="heading-1">{{ analysis.entities.items()|map('join', ': ')|map('capitalize')|join(', ') }}</summary>
        {{ analysis.warning }}
        <img src="{{ analysis.design_matrix }}" />
        <h4>Correlation matrix</h4>
        {% if loop.first %}
        <p>The correlation matrix of a design matrix shows the correlation between
        each pair of regressors. Very high or low correlations among variables of
        interest (top left) or between variables of interest and nuisance regressors
        (top right) can indicate deficiency in the design. High correlations among
        nuisance regressors will generally have little effect on the model.
        </p>
        {% endif %}
        <img src="{{ analysis.correlation_matrix }}" />
        </details>
        {% if step.analyses|count > 1 and loop.first %}
        <details>
        <summary><em>...</em></summary>
        {% endif %}
        {% endfor %}
        </details>
        <!-- { % endif %} -->

        <h3>Contrasts</h3>
        <p>A contrast matrix was generated for each {{ step.name }}. Except
        in very rare cases, these should be identical, so these should be
        inspected to ensure no unexpected differences are present.</p>
        {% for analysis in step.analyses %}
        <details{% if loop.first %} open{% endif %}>
        <summary class="heading-1">{{ analysis.entities.items()|map('join', ': ')|map('capitalize')|join(', ') }}</summary>
        <img src="{{ analysis.contrast_matrix }}" />
        </details>
        {% if step.analyses|count > 1 and loop.first %}
        <details>
        <summary><em>...</em></summary>
        {% endif %}
        {% endfor %}
        </details>

        <!-- { % endfor %} -->
    </div>
    <div id="contrasts">
        <h1 class="sub-report-title">Contrasts</h1>

        {% for step in steps %}
        <h2>{{ step.name|capitalize }} level</h2>
        {% for analysis in step.analyses %}
        <h3>{{ analysis.entities.items()|map('join', ': ')|map('capitalize')|join(', ') }}</h3>
        {% for contrast in analysis.contrasts %}
        <h4>{{ contrast.name }}</h4>
        {% if contrast.glassbrain is none %}
        <p> Missing contrast skipped (used: <code>--drop-missing</code>) </p>
        {% else %}
        <img src="{{ contrast.glassbrain }}" />
        {% endif %}
        {% endfor %}
        {% endfor %}
        {% endfor %}
    </div>
    <div id="about">
        <h1 class="sub-report-title">About</h1>
        <ul>
            <li>Fitlins version: {{ version }}</li>
            <li>Fitlins command: <tt>{{ command }}</tt></li>
            <li>Date processed: {{ timestamp }}</li>
        </ul>
    </div>
</body>
</html>
