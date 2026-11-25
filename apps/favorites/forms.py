from django import forms
from apps.matches.models import Team
from .models import FavoriteTeam


class FavoriteTeamForm(forms.ModelForm):
    class Meta:
        model = FavoriteTeam
        fields = ["team", "priority", "note"]
        labels = {
            "team": "Équipe",
            "priority": "Priorité",
            "note": "Note (optionnelle)",
        }
        widgets = {
            "team": forms.Select(attrs={"class": "form-select"}),
            "priority": forms.Select(attrs={"class": "form-select"}),
            "note": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Ex: À suivre pour le titre...",
                    "maxlength": 200,
                }
            ),
        }

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.fields["team"].queryset = Team.objects.all().order_by("name")

        # Disable team selection on update
        if self.instance and self.instance.pk:
            self.fields["team"].disabled = True

    def clean_team(self):
        team = self.cleaned_data.get("team")
        if not self.instance.pk and self.user and team:
            if FavoriteTeam.objects.filter(user=self.user, team=team).exists():
                raise forms.ValidationError(
                    "Cette équipe fait déjà partie de vos favoris."
                )
        return team