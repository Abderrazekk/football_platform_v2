from django import forms
from .models import Team


class PredictionForm(forms.Form):
    home_team = forms.ModelChoiceField(
        queryset=Team.objects.all().order_by("name"),
        label="Home Team",
        empty_label="Select Home Team",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    away_team = forms.ModelChoiceField(
        queryset=Team.objects.all().order_by("name"),
        label="Away Team",
        empty_label="Select Away Team",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def clean(self):
        cleaned_data = super().clean()
        home = cleaned_data.get("home_team")
        away = cleaned_data.get("away_team")
        if home and away and home == away:
            raise forms.ValidationError("Home team and away team cannot be the same.")
        return cleaned_data