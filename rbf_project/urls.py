from django.contrib import admin
from django.urls import path, include
from django.conf import settings # Importar
from django.conf.urls.static import static # Importar

urlpatterns = [
    path('api/', include('trainer.urls')),
]

# Añadir esto al final
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)