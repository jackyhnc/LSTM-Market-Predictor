using MudBlazor.Services; // mud
using frontend.Components;

var builder = WebApplication.CreateBuilder(args);


// Configure MudBlazor services with options to handle .NET 9 InteractiveServer timing issues
builder.Services.AddMudServices(options =>
{
    // Disable strict provider checking to avoid timing issues with InteractiveServer render mode
    options.PopoverOptions.CheckForPopoverProvider = false;
});

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// apis
builder.Services.AddHttpClient();

builder.Services.AddHttpClient("ModelInferenceBackend", client => {
    client.BaseAddress = new Uri("http://127.0.0.1:8000");
    client.Timeout = TimeSpan.FromMinutes(10);
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
    app.UseHttpsRedirection();
}

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();


