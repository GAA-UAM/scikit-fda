{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      {% if item != "__init__" %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   
   {% for item in methods %}
      {% if item != "__init__" %}
   .. automethod:: {{ item }}
      {% endif %}
   {%- endfor %}
   {% endblock %}
   
.. include:: {{package}}/backreferences/{{fullname}}.examples