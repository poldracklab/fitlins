<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Fitlins - {{ level }}-level report</title>
<style type="text/css">
.sub-report-title {}
.run-title {}
.elem-title {}
.elem-desc {}
.elem-filename {}
.warning {
    border: 1px solid #ffaaaa;
    background: #ffe8e8;
    padding: 0.8em;
}
</style>
</head>
<body>
    <div id="summary">
        <h1 class="sub-report-title">Summary</h1>
        <ul class="elem-desc">
            {% if subject_id %}<li>Subject ID: {{ subject_id }}</li>{% endif %}
            <li>Dataset: {{ dataset }}</li>
            <li>Model: {{ model_name }}</li>
        </ul>
    </div>
    <div id="model">
        <h1 class="sub-report-title">Model</h1>
        {{ warning }}
        {% if design_matrix_svg %}
        <h2>Design matrix</h3>
        <img src="{{ design_matrix_svg }}" />
        {% endif %}
        <h2>Contrasts</h3>
        <img src="{{ contrasts_svg }}" />
        {% if correlation_matrix_svg %}
        <h2>Correlation matrix</h3>
        <img src="{{ correlation_matrix_svg }}" />
        {% endif %}
    </div>
    {% if contrasts %}
    <div id="contrasts">
        <h1 class="sub-report-title">Contrasts</h1>
        {% for contrast in contrasts %}
        <h2>{{ contrast.name }}</h2>
        <img class="ortho" src="{{ contrast.image_file }}" />
        {% endfor %}
    </div>
    {% endif %}
    {% if estimates %}
    <div id="estimates">
        <h1 class="sub-report-title">Estimates</h1>
        {% for estimate in estimates %}
        <h2>{{ estimate.name }}</h2>
        <img class="ortho" src="{{ estimate.image_file }}" />
        {% endfor %}
    </div>
    {% endif %}
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
